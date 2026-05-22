// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/encode.h>
#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>

/* Stacking JXL and Gzip content encoding is not useful. Consequently,
   it is almost legal to reuse this "buffered" bit - especially given that
   subsequent filters are only called once the bit is cleared. */
#define NGX_HTTP_JXL_BUFFERED NGX_HTTP_GZIP_BUFFERED

/* Module configuration. */
typedef struct {
  ngx_flag_t enable;
} ngx_http_jxl_conf_t;

#define BUF_SIZE (1 << 12)

/* Instance context. */
typedef struct {
  /* Input buffer, flattened. TODO: avoid this. */
  u_char *in;
  size_t in_received;
  size_t in_total;

  u_char buf[BUF_SIZE];
  ngx_buf_t out_buf;
  size_t total_out;

  u_char is_writing_out : 1;
  u_char is_err : 1;
  u_char is_done : 1;

  JxlEncoder *encoder;
  JxlEncoderOptions *options;

  ngx_http_request_t *request;
} ngx_http_jxl_ctx_t;

/* Forward declarations. */
static void *ngx_http_jxl_create_conf(ngx_conf_t *cf);
static char *ngx_http_jxl_merge_conf(ngx_conf_t *cf, void *parent, void *child);
static ngx_int_t ngx_http_jxl_filter_init(ngx_conf_t *cf);

/* Configuration literals. */

static ngx_command_t ngx_http_jxl_filter_commands[] = {
    {ngx_string("jxl"),
     NGX_HTTP_MAIN_CONF | NGX_HTTP_SRV_CONF | NGX_HTTP_LOC_CONF |
         NGX_HTTP_LIF_CONF | NGX_CONF_FLAG,
     ngx_conf_set_flag_slot, NGX_HTTP_LOC_CONF_OFFSET,
     offsetof(ngx_http_jxl_conf_t, enable), NULL},

    ngx_null_command};

/* Module context hooks. */
static ngx_http_module_t ngx_http_jxl_filter_module_ctx = {
    NULL,                     /* pre-configuration */
    ngx_http_jxl_filter_init, /* post-configuration */

    NULL, /* create main configuration */
    NULL, /* init main configuration */

    NULL, /* create server configuration */
    NULL, /* merge server configuration */

    ngx_http_jxl_create_conf, /* create location configuration */
    ngx_http_jxl_merge_conf   /* merge location configuration */
};

/* Module descriptor. */
ngx_module_t ngx_http_jxl_filter_module = {
    NGX_MODULE_V1,
    &ngx_http_jxl_filter_module_ctx, /* module context */
    ngx_http_jxl_filter_commands,    /* module directives */
    NGX_HTTP_MODULE,                 /* module type */
    NULL,                            /* init master */
    NULL,                            /* init module */
    NULL,                            /* init process */
    NULL,                            /* init thread */
    NULL,                            /* exit thread */
    NULL,                            /* exit process */
    NULL,                            /* exit master */
    NGX_MODULE_V1_PADDING};

/* Next filter in the filter chain. */
static ngx_http_output_header_filter_pt ngx_http_next_header_filter;
static ngx_http_output_body_filter_pt ngx_http_next_body_filter;

void ngx_http_jxl_free_buffers(ngx_http_jxl_ctx_t *ctx) {
  ngx_pfree(ctx->request->pool, ctx->in);
  JxlEncoderDestroy(ctx->encoder);
  ctx->encoder = NULL;
  ctx->in = NULL;
}

static /* const */ char kType[] = "image/jxl";
static const size_t kTypeLen = 9;
static ngx_int_t check_accept_header(ngx_http_request_t *req) {
  ngx_table_elt_t *accept_entry;
  ngx_str_t *accept_header;
  u_char *start_of_type;
  u_char *end;
  u_char before;
  u_char after;

  accept_entry = req->headers_in.accept;
  if (accept_entry == NULL) return NGX_DECLINED;
  accept_header = &accept_entry->value;

  start_of_type = accept_header->data;
  end = start_of_type + accept_header->len;
  while (1) {
    u_char digit;
    start_of_type = ngx_strcasestrn(start_of_type, kType, kTypeLen - 1);
    if (start_of_type == NULL) return NGX_DECLINED;
    before = (start_of_type == accept_header->data) ? ' ' : start_of_type[-1];
    start_of_type += kTypeLen;
    after = (start_of_type >= end) ? ' ' : *start_of_type;
    if (before != ',' && before != ' ') continue;
    if (after != ',' && after != ' ' && after != ';') continue;

    /* Check for ";q=0[.[0[0[0]]]]" */
    while (*start_of_type == ' ') start_of_type++;
    if (*(start_of_type++) != ';') break;
    while (*start_of_type == ' ') start_of_type++;
    if (*(start_of_type++) != 'q') break;
    while (*start_of_type == ' ') start_of_type++;
    if (*(start_of_type++) != '=') break;
    while (*start_of_type == ' ') start_of_type++;
    if (*(start_of_type++) != '0') break;
    if (*(start_of_type++) != '.') return NGX_DECLINED; /* ;q=0, */
    digit = *(start_of_type++);
    if (digit < '0' || digit > '9') return NGX_DECLINED; /* ;q=0., */
    if (digit > '0') break;
    digit = *(start_of_type++);
    if (digit < '0' || digit > '9') return NGX_DECLINED; /* ;q=0.0, */
    if (digit > '0') break;
    digit = *(start_of_type++);
    if (digit < '0' || digit > '9') return NGX_DECLINED; /* ;q=0.00, */
    if (digit > '0') break;
    return NGX_DECLINED; /* ;q=0.000 */
  }
  return NGX_OK;
}

/* Test if this request is allowed to have the jxl response. */
static ngx_int_t check_jxl_eligility(ngx_http_request_t *req) {
  if (req != req->main) return NGX_DECLINED;
  if (check_accept_header(req) != NGX_OK) return NGX_DECLINED;
  return NGX_OK;
}

/* Process headers and decide if request is eligible for jxl lossless
 * recompression. */
static ngx_int_t ngx_http_jxl_header_filter(ngx_http_request_t *r) {
  ngx_table_elt_t *vary_header;
  ngx_http_jxl_ctx_t *ctx;
  ngx_http_jxl_conf_t *conf;

  conf = ngx_http_get_module_loc_conf(r, ngx_http_jxl_filter_module);

  /* Filter only if enabled. */
  if (!conf->enable) {
    return ngx_http_next_header_filter(r);
  }

  /* Only compress OK responses */
  if (r->headers_out.status != NGX_HTTP_OK) {
    return ngx_http_next_header_filter(r);
  }

  /* Bypass "header only" responses. */
  if (r->header_only) {
    return ngx_http_next_header_filter(r);
  }

  /* Bypass content-encoded compressed responses. */
  if (r->headers_out.content_encoding &&
      r->headers_out.content_encoding->value.len) {
    return ngx_http_next_header_filter(r);
  }

  /* TODO: implement unknown response size */
  if (r->headers_out.content_length_n == -1) {
    return ngx_http_next_header_filter(r);
  }

  // Only compress JPEG files.
  if (ngx_strcasecmp(r->headers_out.content_type.data,
                     (u_char *)"image/jpeg") != 0 &&
      ngx_strcasecmp(r->headers_out.content_type.data, (u_char *)"image/jpg") !=
          0) {
    return ngx_http_next_header_filter(r);
  }

  // Append "Vary" header.
  vary_header = ngx_list_push(&r->headers_out.headers);
  if (vary_header == NULL) {
    return NGX_ERROR;
  }

  vary_header->hash = 1;
  ngx_str_set(&vary_header->key, "Vary");
  ngx_str_set(&vary_header->value, "Accept");

  /* Check if client support jxl encoding. */
  if (check_jxl_eligility(r) != NGX_OK) {
    return ngx_http_next_header_filter(r);
  }

  /* Prepare instance context. */
  ctx = ngx_pcalloc(r->pool, sizeof(ngx_http_jxl_ctx_t));
  if (ctx == NULL) {
    return NGX_ERROR;
  }
  ctx->request = r;
  ctx->in_total = r->headers_out.content_length_n;
  ctx->in = ngx_pcalloc(r->pool, ctx->in_total);
  ngx_http_set_ctx(r, ctx, ngx_http_jxl_filter_module);

  ctx->encoder = JxlEncoderCreate(NULL);
  if (!ctx->encoder) {
    ngx_log_error(NGX_LOG_ALERT, r->connection->log, 0,
                  "JxlEncoderCreate failed");
    ngx_http_jxl_free_buffers(ctx);
    return NGX_ERROR;
  }

  if (JxlEncoderUseContainer(ctx->encoder, JXL_FALSE) != JXL_ENC_SUCCESS) {
    ngx_log_error(NGX_LOG_ALERT, r->connection->log, 0,
                  "JxlEncoderUseContainer failed");
    return NGX_ERROR;
  }

  if (JxlEncoderStoreJPEGMetadata(ctx->encoder, JXL_FALSE) != JXL_ENC_SUCCESS) {
    ngx_log_error(NGX_LOG_ALERT, r->connection->log, 0,
                  "JxlEncoderStoreJPEGMetadata failed");
    return NGX_ERROR;
  }

  ctx->options = JxlEncoderOptionsCreate(ctx->encoder, NULL);

  if (ctx->options == NULL) {
    ngx_log_error(NGX_LOG_ALERT, r->connection->log, 0,
                  "JxlEncoderOptionsCreate failed");
    return NGX_ERROR;
  }

  /* Set content type header. */
  ngx_str_set(&r->headers_out.content_type, "image/jxl");
  r->headers_out.content_type.len = 9;

  r->main_filter_need_in_memory = 1;

  ngx_http_clear_content_length(r);
  ngx_http_clear_accept_ranges(r);
  ngx_http_weak_etag(r);

  return ngx_http_next_header_filter(r);
}

/* Response body filtration (recompression). */
static ngx_int_t ngx_http_jxl_body_filter(ngx_http_request_t *r,
                                          ngx_chain_t *in) {
  ngx_http_jxl_ctx_t *ctx;
  JxlEncoderStatus status;
  ngx_chain_t out_chain;
  ngx_int_t rc;
  uint8_t *out_data;
  size_t out_chunk;

  ctx = ngx_http_get_module_ctx(r, ngx_http_jxl_filter_module);

  ngx_log_debug0(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, "http jxl filter");

  if (ctx == NULL || ctx->in == NULL || r->header_only) {
    return ngx_http_next_body_filter(r, in);
  }

  for (; in; in = in->next) {
    ngx_memcpy(ctx->in + ctx->in_received, in->buf->pos,
               in->buf->last - in->buf->pos);
    ctx->in_received += in->buf->last - in->buf->pos;
    in->buf->pos = in->buf->last;
    if (ctx->in_received < ctx->in_total && in->buf->last_buf) {
      ngx_http_jxl_free_buffers(ctx);
      return NGX_ERROR;
    }
  }

  if (ctx->in_received > ctx->in_total) {
    ngx_http_jxl_free_buffers(ctx);
    return NGX_ERROR; /* Ought to be impossible. */
  }

  // Get more input.
  if (ctx->in_received < ctx->in_total) {
    r->connection->buffered |= NGX_HTTP_JXL_BUFFERED;
    return NGX_OK;
  }

  r->connection->buffered &= ~NGX_HTTP_JXL_BUFFERED;

  if (!ctx->is_writing_out) {
    if (JxlEncoderAddJPEGFrame(ctx->options, ctx->in, ctx->in_total) !=
        JXL_ENC_SUCCESS) {
      ngx_log_error(NGX_LOG_ALERT, r->connection->log, 0,
                    "JxlEncoderAddJPEGFrame failed");
      ctx->is_err = 1;
    }

    JxlEncoderCloseInput(ctx->encoder);
  }

  out_chain.buf = &ctx->out_buf;
  out_chain.next = NULL;

  if (ctx->is_err) {
    if (!ctx->is_writing_out) {
    error:
      ctx->out_buf.pos = ctx->out_buf.start = ctx->in;
      ctx->out_buf.last = ctx->out_buf.end = ctx->in + ctx->in_total;
      ctx->out_buf.last_buf = 1;
      ctx->out_buf.last_in_chain = 1;
      ctx->out_buf.temporary = 1;
      ctx->out_buf.flush = 1;
    }

    for (;;) {
      rc = ngx_http_next_body_filter(r, &out_chain);
      if (rc == NGX_AGAIN) {
        // Will be called again.
        return NGX_AGAIN;
      }
      if (rc == NGX_OK && ctx->out_buf.pos != ctx->out_buf.last) {
        continue;
      }
      ngx_http_jxl_free_buffers(ctx);
      return rc;
    }
  }

  if (ctx->is_writing_out && ctx->out_buf.pos != ctx->out_buf.last) {
    for (;;) {
      rc = ngx_http_next_body_filter(r, &out_chain);
      if (rc == NGX_AGAIN) {
        // Will be called again.
        return NGX_AGAIN;
      }
      if (rc == NGX_OK && ctx->out_buf.pos != ctx->out_buf.last) {
        continue;
      }
      if (rc == NGX_OK && !ctx->is_done) {
        break;
      }
      ngx_http_jxl_free_buffers(ctx);
      return rc;
    }
  }

  ctx->is_writing_out = 1;

  status = JXL_ENC_NEED_MORE_OUTPUT;
  for (;;) {
    out_chunk = BUF_SIZE;
    out_data = ctx->buf;
    status = JxlEncoderProcessOutput(ctx->encoder, &out_data, &out_chunk);
    if (status == JXL_ENC_SUCCESS) {
      ctx->is_done = 1;
    }
    if (status == JXL_ENC_ERROR) {
      if (ctx->total_out == 0) {
        ngx_log_error(NGX_LOG_ALERT, r->connection->log, 0,
                      "JxlEncoderAddJPEGFrame failed");
        ctx->is_err = 1;
        goto error;
      }
      ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                    "JxlEncoderAddJPEGFrame failed after producing output");
      ngx_http_jxl_free_buffers(ctx);
      return NGX_ERROR;
    }
    ctx->total_out += BUF_SIZE - out_chunk;

    ctx->out_buf.pos = ctx->out_buf.start = ctx->buf;
    ctx->out_buf.last = ctx->out_buf.end = out_data;
    ctx->out_buf.last_buf = ctx->is_done;
    ctx->out_buf.last_in_chain = 1;
    ctx->out_buf.temporary = 1;
    ctx->out_buf.flush = 1;

    for (;;) {
      rc = ngx_http_next_body_filter(r, &out_chain);
      if (rc == NGX_AGAIN) {
        // Will be called again.
        return NGX_AGAIN;
      }
      if (rc == NGX_OK && ctx->out_buf.pos != ctx->out_buf.last) {
        continue;
      }
      if (rc == NGX_OK && !ctx->is_done) {
        break;
      }
      ngx_http_jxl_free_buffers(ctx);
      return rc;
    }
  }

  // Unreachable.
  ngx_http_jxl_free_buffers(ctx);
  return NGX_ERROR;
}

static void *ngx_http_jxl_create_conf(ngx_conf_t *cf) {
  ngx_http_jxl_conf_t *conf;
  conf = ngx_pcalloc(cf->pool, sizeof(ngx_http_jxl_conf_t));
  if (conf == NULL) {
    return NULL;
  }
  conf->enable = NGX_CONF_UNSET;
  return conf;
}

static char *ngx_http_jxl_merge_conf(ngx_conf_t *cf, void *parent,
                                     void *child) {
  ngx_http_jxl_conf_t *prev = parent;
  ngx_http_jxl_conf_t *conf = child;
  ngx_conf_merge_value(conf->enable, prev->enable, 0);
  return NGX_CONF_OK;
}

/* Prepend to filter chain. */
static ngx_int_t ngx_http_jxl_filter_init(ngx_conf_t *cf) {
  ngx_http_next_header_filter = ngx_http_top_header_filter;
  ngx_http_top_header_filter = ngx_http_jxl_header_filter;

  ngx_http_next_body_filter = ngx_http_top_body_filter;
  ngx_http_top_body_filter = ngx_http_jxl_body_filter;

  return NGX_OK;
}
