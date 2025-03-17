import numpy as np
import dpnp

def crc16(data, poly=0x8408):
    '''
    CRC-16-CCITT Algorithm using dpnp
    '''
    crc = 0xFFFF
    
    # Convert the data to a dpnp array for device execution
    data_np = np.asarray(data, dtype=np.int32)
    
    for b in data_np:
        cur_byte = 0xFF & b
        
        for _ in range(0, 8):
            if (crc & 0x0001) ^ (cur_byte & 0x0001):
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
            cur_byte >>= 1
    
    crc = (~crc & 0xFFFF)
    crc = (crc << 8) | ((crc >> 8) & 0xFF)
    
    return crc & 0xFFFF
