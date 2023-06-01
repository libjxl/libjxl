// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jpegli/entropy_coding.h"

#include "lib/jpegli/encode_internal.h"
#include "lib/jpegli/error.h"
#include "lib/jpegli/huffman.h"
#include "lib/jxl/base/bits.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jpegli/entropy_coding.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jpegli/entropy_coding-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jpegli {
namespace HWY_NAMESPACE {

void ComputeTokensSequential(const coeff_t* block, int last_dc, int histo_dc,
                             int histo_ac, Token** tokens_ptr) {
  ComputeTokensForBlock<coeff_t, true>(block, last_dc, histo_dc, histo_ac,
                                       tokens_ptr);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jpegli
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jpegli {
namespace {
HWY_EXPORT(ComputeTokensSequential);

float HistogramCost(const Histogram& histo) {
  std::vector<uint32_t> counts(kJpegHuffmanAlphabetSize + 1);
  std::vector<uint8_t> depths(kJpegHuffmanAlphabetSize + 1);
  for (size_t i = 0; i < kJpegHuffmanAlphabetSize; ++i) {
    counts[i] = histo.count[i];
  }
  counts[kJpegHuffmanAlphabetSize] = 1;
  CreateHuffmanTree(counts.data(), counts.size(), kJpegHuffmanMaxBitLength,
                    &depths[0]);
  size_t header_bits = (1 + kJpegHuffmanMaxBitLength) * 8;
  size_t data_bits = 0;
  for (size_t i = 0; i < kJpegHuffmanAlphabetSize; ++i) {
    if (depths[i] > 0) {
      header_bits += 8;
      data_bits += counts[i] * depths[i];
    }
  }
  return header_bits + data_bits;
}

void AddHistograms(const Histogram& a, const Histogram& b, Histogram* c) {
  for (size_t i = 0; i < kJpegHuffmanAlphabetSize; ++i) {
    c->count[i] = a.count[i] + b.count[i];
  }
}

bool IsEmptyHistogram(const Histogram& histo) {
  for (size_t i = 0; i < kJpegHuffmanAlphabetSize; ++i) {
    if (histo.count[i]) return false;
  }
  return true;
}

}  // namespace

void ClusterJpegHistograms(const Histogram* histograms, size_t num,
                           JpegClusteredHistograms* clusters) {
  clusters->histogram_indexes.resize(num);
  std::vector<uint32_t> slot_histograms;
  std::vector<float> slot_costs;
  for (size_t i = 0; i < num; ++i) {
    const Histogram& cur = histograms[i];
    if (IsEmptyHistogram(cur)) {
      continue;
    }
    float best_cost = HistogramCost(cur);
    size_t best_slot = slot_histograms.size();
    for (size_t j = 0; j < slot_histograms.size(); ++j) {
      size_t prev_idx = slot_histograms[j];
      const Histogram& prev = clusters->histograms[prev_idx];
      Histogram combined;
      AddHistograms(prev, cur, &combined);
      float combined_cost = HistogramCost(combined);
      float cost = combined_cost - slot_costs[j];
      if (cost < best_cost) {
        best_cost = cost;
        best_slot = j;
      }
    }
    if (best_slot == slot_histograms.size()) {
      // Create new histogram.
      size_t histogram_index = clusters->histograms.size();
      clusters->histograms.push_back(cur);
      clusters->histogram_indexes[i] = histogram_index;
      if (best_slot < 4) {
        // We have a free slot, so we put the new histogram there.
        slot_histograms.push_back(histogram_index);
        slot_costs.push_back(best_cost);
      } else {
        // TODO(szabadka) Find the best histogram to replce.
        best_slot = (clusters->slot_ids.back() + 1) % 4;
      }
      slot_histograms[best_slot] = histogram_index;
      slot_costs[best_slot] = best_cost;
      clusters->slot_ids.push_back(best_slot);
    } else {
      // Merge this histogram with a previous one.
      size_t histogram_index = slot_histograms[best_slot];
      const Histogram& prev = clusters->histograms[histogram_index];
      AddHistograms(prev, cur, &clusters->histograms[histogram_index]);
      clusters->histogram_indexes[i] = histogram_index;
      JXL_ASSERT(clusters->slot_ids[histogram_index] == best_slot);
      slot_costs[best_slot] += best_cost;
    }
  }
}

void BuildJpegHuffmanCode(const Histogram& histo, JPEGHuffmanCode* huff) {
  std::vector<uint32_t> counts(kJpegHuffmanAlphabetSize + 1);
  std::vector<uint8_t> depths(kJpegHuffmanAlphabetSize + 1);
  for (size_t j = 0; j < kJpegHuffmanAlphabetSize; ++j) {
    counts[j] = histo.count[j];
  }
  counts[kJpegHuffmanAlphabetSize] = 1;
  CreateHuffmanTree(counts.data(), counts.size(), kJpegHuffmanMaxBitLength,
                    &depths[0]);
  std::fill(std::begin(huff->counts), std::end(huff->counts), 0);
  std::fill(std::begin(huff->values), std::end(huff->values), 0);
  for (size_t i = 0; i <= kJpegHuffmanAlphabetSize; ++i) {
    if (depths[i] > 0) {
      ++huff->counts[depths[i]];
    }
  }
  int offset[kJpegHuffmanMaxBitLength + 1] = {0};
  for (size_t i = 1; i <= kJpegHuffmanMaxBitLength; ++i) {
    offset[i] = offset[i - 1] + huff->counts[i - 1];
  }
  for (size_t i = 0; i <= kJpegHuffmanAlphabetSize; ++i) {
    if (depths[i] > 0) {
      huff->values[offset[depths[i]]++] = i;
    }
  }
}

void AddJpegHuffmanCode(const Histogram& histogram, size_t slot_id,
                        JPEGHuffmanCode* huff_codes, size_t* num_huff_codes) {
  JPEGHuffmanCode huff_code = {};
  huff_code.slot_id = slot_id;
  BuildJpegHuffmanCode(histogram, &huff_code);
  memcpy(&huff_codes[*num_huff_codes], &huff_code, sizeof(huff_code));
  ++(*num_huff_codes);
}

namespace {
void SetJpegHuffmanCode(const JpegClusteredHistograms& clusters,
                        size_t histogram_id, size_t slot_id_offset,
                        std::vector<uint32_t>& slot_histograms,
                        uint32_t* slot_id, bool* is_baseline,
                        JPEGHuffmanCode* huff_codes, size_t* num_huff_codes) {
  JXL_ASSERT(histogram_id < clusters.histogram_indexes.size());
  uint32_t histogram_index = clusters.histogram_indexes[histogram_id];
  uint32_t id = clusters.slot_ids[histogram_index];
  if (id > 1) {
    *is_baseline = false;
  }
  *slot_id = id + (slot_id_offset / 4);
  if (slot_histograms[id] != histogram_index) {
    AddJpegHuffmanCode(clusters.histograms[histogram_index],
                       slot_id_offset + id, huff_codes, num_huff_codes);
    slot_histograms[id] = histogram_index;
  }
}

void TokenizeProgressiveDC(const coeff_t* coeffs, int histo_idx, int Al,
                           coeff_t* last_dc_coeff, Token** next_token) {
  coeff_t temp2;
  coeff_t temp;
  temp2 = coeffs[0] >> Al;
  temp = temp2 - *last_dc_coeff;
  *last_dc_coeff = temp2;
  temp2 = temp;
  if (temp < 0) {
    temp = -temp;
    temp2--;
  }
  int nbits = (temp == 0) ? 0 : (jxl::FloorLog2Nonzero<uint32_t>(temp) + 1);
  int bits = temp2 & ((1 << nbits) - 1);
  *(*next_token)++ = Token(histo_idx, nbits, bits);
}

void TokenizeACProgressiveScan(j_compress_ptr cinfo, int scan_index,
                               int histo_idx, ScanTokenInfo* sti) {
  jpeg_comp_master* m = cinfo->master;
  const jpeg_scan_info* scan_info = &cinfo->scan_info[scan_index];
  const int comp_idx = scan_info->component_index[0];
  const jpeg_component_info* comp = &cinfo->comp_info[comp_idx];
  const int Al = scan_info->Al;
  const int Ss = scan_info->Ss;
  const int Se = scan_info->Se;
  const size_t restart_interval = sti->restart_interval;
  int restarts_to_go = restart_interval;
  size_t num_blocks = comp->height_in_blocks * comp->width_in_blocks;
  size_t num_restarts =
      restart_interval > 0 ? DivCeil(num_blocks, restart_interval) : 1;
  size_t restart_idx = 0;
  int eob_run = 0;
  TokenArray* ta = &m->token_arrays[m->cur_token_array];
  sti->token_offset = m->total_num_tokens + ta->num_tokens;
  sti->restarts = Allocate<size_t>(cinfo, num_restarts, JPOOL_IMAGE);
  for (JDIMENSION by = 0; by < comp->height_in_blocks; ++by) {
    JBLOCKARRAY ba = (*cinfo->mem->access_virt_barray)(
        reinterpret_cast<j_common_ptr>(cinfo), m->coeff_buffers[comp_idx], by,
        1, false);
    int max_tokens_per_row = comp->width_in_blocks * (Se - Ss + 1);
    if (ta->num_tokens + max_tokens_per_row > m->num_tokens) {
      if (ta->tokens) {
        m->total_num_tokens += ta->num_tokens;
        ++m->cur_token_array;
        ta = &m->token_arrays[m->cur_token_array];
      }
      m->num_tokens =
          EstimateNumTokens(cinfo, by, comp->height_in_blocks,
                            m->total_num_tokens, max_tokens_per_row);
      ta->tokens = Allocate<Token>(cinfo, m->num_tokens, JPOOL_IMAGE);
      m->next_token = ta->tokens;
    }
    for (JDIMENSION bx = 0; bx < comp->width_in_blocks; ++bx) {
      if (restart_interval > 0 && restarts_to_go == 0) {
        if (eob_run > 0) {
          int nbits = jxl::FloorLog2Nonzero<uint32_t>(eob_run);
          int symbol = nbits << 4u;
          *m->next_token++ =
              Token(histo_idx, symbol, eob_run & ((1 << nbits) - 1));
          eob_run = 0;
        }
        ta->num_tokens = m->next_token - ta->tokens;
        sti->restarts[restart_idx++] = m->total_num_tokens + ta->num_tokens;
        restarts_to_go = restart_interval;
      }
      const coeff_t* block = &ba[0][bx][0];
      coeff_t temp2;
      coeff_t temp;
      int r = 0;
      int num_nzeros = 0;
      int num_future_nzeros = 0;
      for (int k = Ss; k <= Se; ++k) {
        if ((temp = block[k]) == 0) {
          r++;
          continue;
        }
        if (temp < 0) {
          temp = -temp;
          temp >>= Al;
          temp2 = ~temp;
        } else {
          temp >>= Al;
          temp2 = temp;
        }
        if (temp == 0) {
          r++;
          num_future_nzeros++;
          continue;
        }
        if (eob_run > 0) {
          int nbits = jxl::FloorLog2Nonzero<uint32_t>(eob_run);
          int symbol = nbits << 4u;
          *m->next_token++ =
              Token(histo_idx, symbol, eob_run & ((1 << nbits) - 1));
          eob_run = 0;
        }
        while (r > 15) {
          *m->next_token++ = Token(histo_idx, 0xf0, 0);
          r -= 16;
        }
        int nbits = jxl::FloorLog2Nonzero<uint32_t>(temp) + 1;
        int symbol = (r << 4u) + nbits;
        *m->next_token++ = Token(histo_idx, symbol, temp2 & ((1 << nbits) - 1));
        ++num_nzeros;
        r = 0;
      }
      if (r > 0) {
        ++eob_run;
        if (eob_run == 0x7FFF) {
          int nbits = jxl::FloorLog2Nonzero<uint32_t>(eob_run);
          int symbol = nbits << 4u;
          *m->next_token++ =
              Token(histo_idx, symbol, eob_run & ((1 << nbits) - 1));
          eob_run = 0;
        }
      }
      sti->num_nonzeros += num_nzeros;
      sti->num_future_nonzeros += num_future_nzeros;
      --restarts_to_go;
    }
    ta->num_tokens = m->next_token - ta->tokens;
  }
  if (eob_run > 0) {
    int nbits = jxl::FloorLog2Nonzero<uint32_t>(eob_run);
    int symbol = nbits << 4u;
    *m->next_token++ = Token(histo_idx, symbol, eob_run & ((1 << nbits) - 1));
    ++ta->num_tokens;
    eob_run = 0;
  }
  sti->num_tokens = m->total_num_tokens + ta->num_tokens - sti->token_offset;
  sti->restarts[restart_idx++] = m->total_num_tokens + ta->num_tokens;
}

void TokenizeACRefinementScan(j_compress_ptr cinfo, int scan_index,
                              ScanTokenInfo* sti) {
  jpeg_comp_master* m = cinfo->master;
  const jpeg_scan_info* scan_info = &cinfo->scan_info[scan_index];
  const int comp_idx = scan_info->component_index[0];
  const jpeg_component_info* comp = &cinfo->comp_info[comp_idx];
  const int Al = scan_info->Al;
  const int Ss = scan_info->Ss;
  const int Se = scan_info->Se;
  const size_t restart_interval = sti->restart_interval;
  int restarts_to_go = restart_interval;
  RefToken token;
  int eob_run = 0;
  int eob_refbits = 0;
  size_t num_blocks = comp->height_in_blocks * comp->width_in_blocks;
  size_t num_restarts =
      restart_interval > 0 ? DivCeil(num_blocks, restart_interval) : 1;
  sti->tokens = m->next_refinement_token;
  sti->refbits = m->next_refinement_bit;
  sti->eobruns = Allocate<uint16_t>(cinfo, num_blocks / 2, JPOOL_IMAGE);
  sti->restarts = Allocate<size_t>(cinfo, num_restarts, JPOOL_IMAGE);
  RefToken* next_token = sti->tokens;
  RefToken* next_eob_token = next_token;
  uint8_t* next_ref_bit = sti->refbits;
  uint16_t* next_eobrun = sti->eobruns;
  size_t restart_idx = 0;
  for (JDIMENSION by = 0; by < comp->height_in_blocks; ++by) {
    JBLOCKARRAY ba = (*cinfo->mem->access_virt_barray)(
        reinterpret_cast<j_common_ptr>(cinfo), m->coeff_buffers[comp_idx], by,
        1, false);
    for (JDIMENSION bx = 0; bx < comp->width_in_blocks; ++bx) {
      if (restart_interval > 0 && restarts_to_go == 0) {
        sti->restarts[restart_idx++] = next_token - sti->tokens;
        restarts_to_go = restart_interval;
        next_eob_token = next_token;
        eob_run = eob_refbits = 0;
      }
      const coeff_t* block = &ba[0][bx][0];
      int num_eob_refinement_bits = 0;
      int num_refinement_bits = 0;
      int num_nzeros = 0;
      int r = 0;
      for (int k = Ss; k <= Se; ++k) {
        int absval = block[k];
        if (absval == 0) {
          r++;
          continue;
        }
        const int mask = absval >> (8 * sizeof(int) - 1);
        absval += mask;
        absval ^= mask;
        absval >>= Al;
        if (absval == 0) {
          r++;
          continue;
        }
        while (r > 15) {
          token.symbol = 0xf0;
          token.refbits = num_refinement_bits;
          *next_token++ = token;
          r -= 16;
          num_eob_refinement_bits += num_refinement_bits;
          num_refinement_bits = 0;
        }
        if (absval > 1) {
          *next_ref_bit++ = absval & 1u;
          ++num_refinement_bits;
          continue;
        }
        int symbol = (r << 4u) + 1 + ((mask + 1) << 1);
        token.symbol = symbol;
        token.refbits = num_refinement_bits;
        *next_token++ = token;
        ++num_nzeros;
        num_refinement_bits = 0;
        num_eob_refinement_bits = 0;
        r = 0;
        next_eob_token = next_token;
        eob_run = eob_refbits = 0;
      }
      if (r > 0 || num_eob_refinement_bits + num_refinement_bits > 0) {
        ++eob_run;
        eob_refbits += num_eob_refinement_bits + num_refinement_bits;
        if (eob_refbits > 255) {
          ++next_eob_token;
          eob_refbits = num_eob_refinement_bits + num_refinement_bits;
          eob_run = 1;
        }
        next_token = next_eob_token;
        next_token->refbits = eob_refbits;
        if (eob_run == 1) {
          next_token->symbol = 0;
        } else if (eob_run == 2) {
          next_token->symbol = 16;
          *next_eobrun++ = 0;
        } else if ((eob_run & (eob_run - 1)) == 0) {
          next_token->symbol += 16;
          next_eobrun[-1] = 0;
        } else {
          ++next_eobrun[-1];
        }
        ++next_token;
        if (eob_run == 0x7fff) {
          next_eob_token = next_token;
          eob_run = eob_refbits = 0;
        }
      }
      sti->num_nonzeros += num_nzeros;
      --restarts_to_go;
    }
  }
  sti->num_tokens = next_token - sti->tokens;
  sti->restarts[restart_idx++] = sti->num_tokens;
  m->next_refinement_token = next_token;
  m->next_refinement_bit = next_ref_bit;
}

void TokenizeScan(j_compress_ptr cinfo, size_t scan_index, int ac_histo_offset,
                  ScanTokenInfo* sti) {
  const jpeg_scan_info* scan_info = &cinfo->scan_info[scan_index];
  if (scan_info->Ss > 0) {
    if (scan_info->Ah == 0) {
      TokenizeACProgressiveScan(cinfo, scan_index, ac_histo_offset, sti);
    } else {
      TokenizeACRefinementScan(cinfo, scan_index, sti);
    }
    return;
  }

  jpeg_comp_master* m = cinfo->master;
  size_t restart_interval = sti->restart_interval;
  int restarts_to_go = restart_interval;
  coeff_t last_dc_coeff[MAX_COMPS_IN_SCAN] = {0};

  // "Non-interleaved" means color data comes in separate scans, in other words
  // each scan can contain only one color component.
  const bool is_interleaved = (scan_info->comps_in_scan > 1);
  const bool is_progressive = cinfo->progressive_mode;
  const int Ah = scan_info->Ah;
  const int Al = scan_info->Al;
  HWY_ALIGN constexpr coeff_t kDummyBlock[DCTSIZE2] = {0};

  size_t restart_idx = 0;
  TokenArray* ta = &m->token_arrays[m->cur_token_array];
  sti->token_offset = Ah > 0 ? 0 : m->total_num_tokens + ta->num_tokens;

  if (Ah > 0) {
    sti->refbits = Allocate<uint8_t>(cinfo, sti->num_blocks, JPOOL_IMAGE);
  } else if (cinfo->progressive_mode) {
    if (ta->num_tokens + sti->num_blocks > m->num_tokens) {
      if (ta->tokens) {
        m->total_num_tokens += ta->num_tokens;
        ++m->cur_token_array;
        ta = &m->token_arrays[m->cur_token_array];
      }
      m->num_tokens = sti->num_blocks;
      ta->tokens = Allocate<Token>(cinfo, m->num_tokens, JPOOL_IMAGE);
      m->next_token = ta->tokens;
    }
  }

  JBLOCKARRAY ba[MAX_COMPS_IN_SCAN];
  size_t block_idx = 0;
  for (size_t mcu_y = 0; mcu_y < sti->MCU_rows_in_scan; ++mcu_y) {
    for (int i = 0; i < scan_info->comps_in_scan; ++i) {
      int comp_idx = scan_info->component_index[i];
      jpeg_component_info* comp = &cinfo->comp_info[comp_idx];
      int n_blocks_y = is_interleaved ? comp->v_samp_factor : 1;
      int by0 = mcu_y * n_blocks_y;
      int block_rows_left = comp->height_in_blocks - by0;
      int max_block_rows = std::min(n_blocks_y, block_rows_left);
      ba[i] = (*cinfo->mem->access_virt_barray)(
          reinterpret_cast<j_common_ptr>(cinfo), m->coeff_buffers[comp_idx],
          by0, max_block_rows, false);
    }
    if (!cinfo->progressive_mode) {
      int max_tokens_per_mcu_row = MaxNumTokensPerMCURow(cinfo);
      if (ta->num_tokens + max_tokens_per_mcu_row > m->num_tokens) {
        if (ta->tokens) {
          m->total_num_tokens += ta->num_tokens;
          ++m->cur_token_array;
          ta = &m->token_arrays[m->cur_token_array];
        }
        m->num_tokens =
            EstimateNumTokens(cinfo, mcu_y, sti->MCU_rows_in_scan,
                              m->total_num_tokens, max_tokens_per_mcu_row);
        ta->tokens = Allocate<Token>(cinfo, m->num_tokens, JPOOL_IMAGE);
        m->next_token = ta->tokens;
      }
    }
    for (size_t mcu_x = 0; mcu_x < sti->MCUs_per_row; ++mcu_x) {
      // Possibly emit a restart marker.
      if (restart_interval > 0 && restarts_to_go == 0) {
        restarts_to_go = restart_interval;
        memset(last_dc_coeff, 0, sizeof(last_dc_coeff));
        ta->num_tokens = m->next_token - ta->tokens;
        sti->restarts[restart_idx++] =
            Ah > 0 ? block_idx : m->total_num_tokens + ta->num_tokens;
      }
      // Encode one MCU
      for (int i = 0; i < scan_info->comps_in_scan; ++i) {
        int comp_idx = scan_info->component_index[i];
        jpeg_component_info* comp = &cinfo->comp_info[comp_idx];
        int n_blocks_y = is_interleaved ? comp->v_samp_factor : 1;
        int n_blocks_x = is_interleaved ? comp->h_samp_factor : 1;
        for (int iy = 0; iy < n_blocks_y; ++iy) {
          for (int ix = 0; ix < n_blocks_x; ++ix) {
            size_t block_y = mcu_y * n_blocks_y + iy;
            size_t block_x = mcu_x * n_blocks_x + ix;
            const coeff_t* block;
            if (block_x >= comp->width_in_blocks ||
                block_y >= comp->height_in_blocks) {
              block = kDummyBlock;
            } else {
              block = &ba[i][iy][block_x][0];
            }
            if (!is_progressive) {
              HWY_DYNAMIC_DISPATCH(ComputeTokensSequential)
              (block, last_dc_coeff[i], comp_idx, ac_histo_offset + i,
               &m->next_token);
              last_dc_coeff[i] = block[0];
            } else {
              if (Ah == 0) {
                TokenizeProgressiveDC(block, comp_idx, Al, last_dc_coeff + i,
                                      &m->next_token);
              } else {
                sti->refbits[block_idx] = (block[0] >> Al) & 1;
              }
            }
            ++block_idx;
          }
        }
      }
      --restarts_to_go;
    }
    ta->num_tokens = m->next_token - ta->tokens;
  }
  JXL_DASSERT(block_idx == sti->num_blocks);
  sti->num_tokens =
      Ah > 0 ? sti->num_blocks
             : m->total_num_tokens + ta->num_tokens - sti->token_offset;
  sti->restarts[restart_idx++] =
      Ah > 0 ? sti->num_blocks : m->total_num_tokens + ta->num_tokens;
  if (Ah == 0 && cinfo->progressive_mode) {
    JXL_DASSERT(sti->num_blocks == sti->num_tokens);
  }
}

}  // namespace

void TokenizeJpeg(j_compress_ptr cinfo) {
  jpeg_comp_master* m = cinfo->master;
  std::vector<int> processed(cinfo->num_scans);
  size_t max_refinement_tokens = 0;
  size_t num_refinement_bits = 0;
  int num_refinement_scans[DCTSIZE2] = {};
  int max_num_refinement_scans = 0;
  for (int i = 0; i < cinfo->num_scans; ++i) {
    const jpeg_scan_info* si = &cinfo->scan_info[i];
    ScanTokenInfo* sti = &m->scan_token_info[i];
    if (si->Ss > 0 && si->Ah == 0 && si->Al > 0) {
      int offset = m->ac_histogram_offset[i];
      TokenizeScan(cinfo, i, offset, sti);
      processed[i] = 1;
      max_refinement_tokens += sti->num_future_nonzeros;
      for (int k = si->Ss; k <= si->Se; ++k) {
        num_refinement_scans[k] = si->Al;
      }
      max_num_refinement_scans = std::max(max_num_refinement_scans, si->Al);
      num_refinement_bits += sti->num_nonzeros;
    }
    if (si->Ss > 0 && si->Ah > 0) {
      int comp_idx = si->component_index[0];
      const jpeg_component_info* comp = &cinfo->comp_info[comp_idx];
      size_t num_blocks = comp->width_in_blocks * comp->height_in_blocks;
      max_refinement_tokens += (1 + (si->Se - si->Ss) / 16) * num_blocks;
    }
  }
  if (max_refinement_tokens > 0) {
    m->next_refinement_token =
        Allocate<RefToken>(cinfo, max_refinement_tokens, JPOOL_IMAGE);
  }
  for (int j = 0; j < max_num_refinement_scans; ++j) {
    uint8_t* refinement_bits =
        Allocate<uint8_t>(cinfo, num_refinement_bits, JPOOL_IMAGE);
    m->next_refinement_bit = refinement_bits;
    size_t new_refinement_bits = 0;
    for (int i = 0; i < cinfo->num_scans; ++i) {
      const jpeg_scan_info* si = &cinfo->scan_info[i];
      ScanTokenInfo* sti = &m->scan_token_info[i];
      if (si->Ss > 0 && si->Ah > 0 &&
          si->Ah == num_refinement_scans[si->Ss] - j) {
        int offset = m->ac_histogram_offset[i];
        TokenizeScan(cinfo, i, offset, sti);
        processed[i] = 1;
        new_refinement_bits += sti->num_nonzeros;
      }
    }
    JXL_DASSERT(m->next_refinement_bit ==
                refinement_bits + num_refinement_bits);
    num_refinement_bits += new_refinement_bits;
  }
  for (int i = 0; i < cinfo->num_scans; ++i) {
    if (processed[i]) {
      continue;
    }
    int offset = m->ac_histogram_offset[i];
    TokenizeScan(cinfo, i, offset, &m->scan_token_info[i]);
    processed[i] = 1;
  }
}

void CopyHuffmanTable(j_compress_ptr cinfo, int index, bool is_dc,
                      JPEGHuffmanCode* huffman_codes,
                      size_t* num_huffman_codes) {
  const char* type = is_dc ? "DC" : "AC";
  if (index < 0 || index >= NUM_HUFF_TBLS) {
    JPEGLI_ERROR("Invalid %s Huffman table index %d", type, index);
  }
  JHUFF_TBL* table =
      is_dc ? cinfo->dc_huff_tbl_ptrs[index] : cinfo->ac_huff_tbl_ptrs[index];
  if (table == nullptr) {
    JPEGLI_ERROR("Missing %s Huffman table %d", type, index);
  }
  ValidateHuffmanTable(reinterpret_cast<j_common_ptr>(cinfo), table, is_dc);
  JPEGHuffmanCode huff = {};
  size_t max_depth = 0;
  for (size_t i = 1; i <= kJpegHuffmanMaxBitLength; ++i) {
    if (table->bits[i] != 0) max_depth = i;
    huff.counts[i] = table->bits[i];
  }
  ++huff.counts[max_depth];
  for (size_t i = 0; i < kJpegHuffmanAlphabetSize; ++i) {
    huff.values[i] = table->huffval[i];
  }
  huff.slot_id = index + (is_dc ? 0 : 0x10);
  huff.sent_table = table->sent_table;
  bool have_slot = false;
  for (size_t i = 0; i < *num_huffman_codes; ++i) {
    if (huffman_codes[i].slot_id == huff.slot_id) have_slot = true;
  }
  if (!have_slot) {
    memcpy(&huffman_codes[*num_huffman_codes], &huff, sizeof(huff));
    ++(*num_huffman_codes);
  }
}

void CopyHuffmanCodes(j_compress_ptr cinfo, bool* is_baseline) {
  jpeg_comp_master* m = cinfo->master;
  m->huffman_codes =
      Allocate<JPEGHuffmanCode>(cinfo, 2 * cinfo->num_components, JPOOL_IMAGE);
  for (int c = 0; c < cinfo->num_components; ++c) {
    jpeg_component_info* comp = &cinfo->comp_info[c];
    if (comp->dc_tbl_no > 1 || comp->ac_tbl_no > 1) {
      *is_baseline = false;
    }
    CopyHuffmanTable(cinfo, comp->dc_tbl_no, /*is_dc=*/true, m->huffman_codes,
                     &m->num_huffman_codes);
    CopyHuffmanTable(cinfo, comp->ac_tbl_no, /*is_dc=*/false, m->huffman_codes,
                     &m->num_huffman_codes);
  }
  m->context_map = Allocate<uint8_t>(cinfo, 8, JPOOL_IMAGE);
  memset(m->context_map, 0, 8);
  size_t ac_histogram_id = 0;
  for (int i = 0; i < cinfo->num_scans; ++i) {
    const jpeg_scan_info* si = &cinfo->scan_info[i];
    ScanCodingInfo sci = {};
    for (int j = 0; j < si->comps_in_scan; ++j) {
      int ci = si->component_index[j];
      sci.dc_tbl_idx[j] = cinfo->comp_info[ci].dc_tbl_no;
      sci.ac_tbl_idx[j] = cinfo->comp_info[ci].ac_tbl_no + 4;
      m->context_map[ci] = sci.dc_tbl_idx[j];
      m->context_map[4 + ac_histogram_id] = sci.ac_tbl_idx[j];
      ++ac_histogram_id;
    }
    if (i == 0) {
      sci.num_huffman_codes = m->num_huffman_codes;
    }
    memcpy(&m->scan_coding_info[i], &sci, sizeof(sci));
  }
}

size_t MaxNumTokensPerMCURow(j_compress_ptr cinfo) {
  int MCUs_per_row = DivCeil(cinfo->image_width, 8 * cinfo->max_h_samp_factor);
  size_t blocks_per_mcu = 0;
  for (int c = 0; c < cinfo->num_components; ++c) {
    jpeg_component_info* comp = &cinfo->comp_info[c];
    blocks_per_mcu += comp->h_samp_factor * comp->v_samp_factor;
  }
  return kDCTBlockSize * blocks_per_mcu * MCUs_per_row;
}

size_t EstimateNumTokens(j_compress_ptr cinfo, size_t mcu_y, size_t ysize_mcus,
                         size_t num_tokens, size_t max_per_row) {
  size_t estimate;
  if (mcu_y == 0) {
    estimate = 16 * max_per_row;
  } else {
    estimate = (4 * ysize_mcus * num_tokens) / (3 * mcu_y);
  }
  size_t mcus_left = ysize_mcus - mcu_y;
  return std::min(mcus_left * max_per_row,
                  std::max(max_per_row, estimate - num_tokens));
}

size_t RestartIntervalForScan(j_compress_ptr cinfo, size_t scan_index) {
  if (cinfo->restart_in_rows <= 0) {
    return cinfo->restart_interval;
  } else {
    const jpeg_scan_info* scan_info = &cinfo->scan_info[scan_index];
    const bool is_interleaved = (scan_info->comps_in_scan > 1);
    jpeg_component_info* base_comp =
        &cinfo->comp_info[scan_info->component_index[0]];
    const int h_group = is_interleaved ? 1 : base_comp->h_samp_factor;
    int MCUs_per_row =
        DivCeil(cinfo->image_width * h_group, 8 * cinfo->max_h_samp_factor);
    return std::min<size_t>(MCUs_per_row * cinfo->restart_in_rows, 65535u);
  }
}

void BuildHistograms(j_compress_ptr cinfo, Histogram* histograms) {
  jpeg_comp_master* m = cinfo->master;
  size_t num_token_arrays = m->cur_token_array + 1;
  for (size_t i = 0; i < num_token_arrays; ++i) {
    Token* tokens = m->token_arrays[i].tokens;
    size_t num_tokens = m->token_arrays[i].num_tokens;
    for (size_t j = 0; j < num_tokens; ++j) {
      Token t = tokens[j];
      ++histograms[t.histo_idx].count[t.symbol];
    }
  }
  for (int i = 0; i < cinfo->num_scans; ++i) {
    const jpeg_scan_info& si = cinfo->scan_info[i];
    const ScanTokenInfo& sti = m->scan_token_info[i];
    if (si.Ss > 0 && si.Ah > 0) {
      int* ac_histo = &histograms[m->ac_histogram_offset[i]].count[0];
      for (size_t j = 0; j < sti.num_tokens; ++j) {
        ++ac_histo[sti.tokens[j].symbol & 253];
      }
    }
  }
}

void OptimizeHuffmanCodes(j_compress_ptr cinfo, bool* is_baseline) {
  jpeg_comp_master* m = cinfo->master;
  std::vector<Histogram> histograms(m->num_histograms);
  BuildHistograms(cinfo, &histograms[0]);

  // Cluster DC histograms.
  JpegClusteredHistograms dc_clusters;
  ClusterJpegHistograms(histograms.data(), cinfo->num_components, &dc_clusters);

  // Cluster AC histograms.
  JpegClusteredHistograms ac_clusters;
  ClusterJpegHistograms(histograms.data() + 4, m->num_histograms - 4,
                        &ac_clusters);

  // Add the first 4 DC and AC histograms in the first DHT segment.
  std::vector<uint32_t> dc_slot_histograms;
  std::vector<uint32_t> ac_slot_histograms;
  size_t num_histo = m->num_histograms;
  m->huffman_codes = Allocate<JPEGHuffmanCode>(cinfo, num_histo, JPOOL_IMAGE);
  for (size_t i = 0; i < dc_clusters.histograms.size(); ++i) {
    JXL_ASSERT(dc_clusters.slot_ids[i] == i);
    AddJpegHuffmanCode(dc_clusters.histograms[i], i, m->huffman_codes,
                       &m->num_huffman_codes);
    dc_slot_histograms.push_back(i);
  }
  for (size_t i = 0; i < ac_clusters.histograms.size(); ++i) {
    if (i >= 4) break;
    JXL_ASSERT(ac_clusters.slot_ids[i] == i);
    AddJpegHuffmanCode(ac_clusters.histograms[i], 0x10 + i, m->huffman_codes,
                       &m->num_huffman_codes);
    ac_slot_histograms.push_back(i);
  }

  // Set the Huffman table indexes in the scan_infos and emit additional DHT
  // segments if necessary.
  size_t ac_histogram_id = 0;
  size_t num_huffman_codes_sent = 0;
  m->context_map = Allocate<uint8_t>(cinfo, m->num_histograms, JPOOL_IMAGE);
  memset(m->context_map, 0, m->num_histograms);
  for (int i = 0; i < cinfo->num_scans; ++i) {
    ScanCodingInfo sci = {};
    for (int j = 0; j < cinfo->scan_info[i].comps_in_scan; ++j) {
      if (cinfo->scan_info[i].Ss == 0) {
        uint32_t dc_histogram_id = cinfo->scan_info[i].component_index[j];
        SetJpegHuffmanCode(dc_clusters, dc_histogram_id, 0, dc_slot_histograms,
                           &sci.dc_tbl_idx[j], is_baseline, m->huffman_codes,
                           &m->num_huffman_codes);
        m->context_map[dc_histogram_id] = sci.dc_tbl_idx[j];
      }
      if (cinfo->scan_info[i].Se > 0) {
        SetJpegHuffmanCode(ac_clusters, ac_histogram_id, 0x10,
                           ac_slot_histograms, &sci.ac_tbl_idx[j], is_baseline,
                           m->huffman_codes, &m->num_huffman_codes);
        m->context_map[4 + ac_histogram_id] = sci.ac_tbl_idx[j];
        ++ac_histogram_id;
      } else {
        sci.ac_tbl_idx[j] = 4;
      }
    }
    sci.num_huffman_codes = m->num_huffman_codes - num_huffman_codes_sent;
    num_huffman_codes_sent = m->num_huffman_codes;
    memcpy(&m->scan_coding_info[i], &sci, sizeof(sci));
  }
}

}  // namespace jpegli
#endif  // HWY_ONCE
