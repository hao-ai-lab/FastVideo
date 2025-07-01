    off_hz = tl.program_id(2)
    b = off_hz // H
    h = off_hz % H
    meta_base = ((b * H + h) * q_tiles + q_blk)