#### Table M.3 – Protocol Buffer descriptor of top-level structure of losslessly compressed JPEG stream

```protobuf
message Header {
  optional uint64 width = 1;
  optional uint64 height = 2;
  required uint64 version_and_component_count_code = 3;
  optional uint64 subsampling_code = 4;
}

message Jpeg {
  required bytes signature = 1;
  required Header header = 2;
  optional bytes meta_data = 3;
  optional bytes jpeg1_internals = 4;
  optional bytes quant_data = 5;
  optional bytes histogram_data = 6;
  optional bytes dc_data = 7;
  optional bytes ac_data = 8;
  optional bytes original_jpg = 9;
}
```

