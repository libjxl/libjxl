#include <jxl/decode.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

int main() {
    JxlDecoder* dec = JxlDecoderCreate(NULL);
    if (!dec) return 1;

    JxlDecoderSubscribeEvents(dec, JXL_DEC_BOX | JXL_DEC_BOX_NEED_MORE_OUTPUT);
    
    // We mock the JXL file. We just need to feed it a JXL container with a large jhgm box.
    // JXL container signature: 00 00 00 0c 4a 58 4c 20 0d 0a 87 0a
    uint8_t sig[] = {0x00, 0x00, 0x00, 0x0c, 'J', 'X', 'L', ' ', 0x0d, 0x0a, 0x87, 0x0a};
    
    // jhgm box: size = 10000, type = 'jhgm', data...
    uint8_t box_header[] = {
        0x00, 0x00, 0x27, 0x18, // size = 10008
        'j', 'h', 'g', 'm'
    };
    
    size_t data_size = sizeof(sig) + sizeof(box_header);
    uint8_t* data = (uint8_t*)malloc(data_size);
    memcpy(data, sig, sizeof(sig));
    memcpy(data + sizeof(sig), box_header, sizeof(box_header));
    
    JxlDecoderSetInput(dec, data, data_size);
    
    const size_t chunk_size = 2048;
    uint8_t* box_data = NULL;
    size_t box_size = 0;
    size_t box_index = 0;
    
    while (1) {
        JxlDecoderStatus status = JxlDecoderProcessInput(dec);
        if (status == JXL_DEC_ERROR || status == JXL_DEC_SUCCESS) break;
        
        if (status == JXL_DEC_BOX) {
            JxlBoxType box_type;
            JxlDecoderGetBoxType(dec, box_type, JXL_FALSE);
            if (!strncmp(box_type, "jhgm", 4)) {
                box_data = (uint8_t*)malloc(chunk_size);
                box_size = chunk_size;
                JxlDecoderSetBoxBuffer(dec, box_data, box_size);
                printf("Init box buffer: size %zu\n", box_size);
            }
        } else if (status == JXL_DEC_BOX_NEED_MORE_OUTPUT) {
            const size_t remaining = JxlDecoderReleaseBoxBuffer(dec);
            printf("NEED_MORE_OUTPUT: remaining = %zu\n", remaining);
            
            box_size += chunk_size;
            box_index += chunk_size - remaining;
            
            printf("  New box_size: %zu, chunk_size - remaining: %zu, new box_index: %zu\n", box_size, chunk_size - remaining, box_index);
            
            uint8_t* temp = (uint8_t*)realloc(box_data, box_size);
            if (!temp) break;
            box_data = temp;
            
            JxlDecoderSetBoxBuffer(dec, box_data + box_index, box_size - box_index);
            printf("  SetBoxBuffer: ptr offset %zu, size %zu\n", box_index, box_size - box_index);
            
            // To simulate file reading, we should feed more input, but for this test we only want to see if remaining > chunk_size can happen without new input?
            // Actually, NEED_MORE_OUTPUT means the decoder needs more OUTPUT space (i.e. our box buffer was too small or it wants us to process the buffer).
            // But wait! `JxlDecoderSetInput` is for input data. `JxlDecoderSetBoxBuffer` is for *output* of the box data.
            // If the box is 10000 bytes, the decoder reads input, sees it's a jhgm box, and starts emitting the box payload into our BoxBuffer.
            // Wait, we didn't give it any payload data yet!
            // Let's feed more dummy payload data.
            uint8_t payload[4096] = {0};
            JxlDecoderSetInput(dec, payload, sizeof(payload));
        } else if (status == JXL_DEC_NEED_MORE_INPUT) {
            printf("NEED_MORE_INPUT\n");
            uint8_t payload[4096] = {0};
            JxlDecoderSetInput(dec, payload, sizeof(payload));
        }
    }
    
    JxlDecoderDestroy(dec);
    free(data);
    if (box_data) free(box_data);
    return 0;
}
