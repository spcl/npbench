import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def crc16(data, poly=0x8408):
    '''
    CRC-16-CCITT Algorithm
    '''
    crc = 0xFFFF
    
    def loop_body(crc, b):
        cur_byte = 0xFF & b

        def inner_loop_body(carry, data):
            crc, cur_byte = carry
            xor_flag = (crc & 0x0001) ^ (cur_byte & 0x0001)
            crc = lax.select(xor_flag, (crc >> 1) ^ poly, crc >> 1)
            cur_byte >>= 1

            return (crc, cur_byte), None
        
        (crc, cur_byte), _ = lax.scan(inner_loop_body, (crc, cur_byte), jnp.arange(8))

        return crc, None
    
    crc, _ = lax.scan(loop_body, crc, data)

    crc = (~crc & 0xFFFF)
    crc = (crc << 8) | ((crc >> 8) & 0xFF)

    
    return crc & 0xFFFF
