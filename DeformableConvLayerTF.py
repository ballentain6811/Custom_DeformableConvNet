import numpy as np
import tensorflow as tf

def deformable_conv(input, name, batch_size, filters, kernel_size):
    input_size = input.get_shape().as_list()[1]

    # Input image 좌표계 관련
    INPUT_GRID = []
    grid_x, grid_y = tf.meshgrid(tf.range(input_size), tf.range(input_size))
    for grid in [grid_x, grid_y]:
        grid = tf.reshape(grid, [1, *grid.get_shape(), 1])
        patched_grid = tf.image.extract_image_patches(grid,
                                                      ksizes = [1, kernel_size, kernel_size, 1],
                                                      strides = [1, 1, 1, 1],
                                                      rates = [1, 1, 1, 1],
                                                      padding = 'SAME')
        batch_patched_grid = tf.tile(patched_grid, [batch_size, 1, 1, 1])
        batch_patched_grid = tf.cast(batch_patched_grid, tf.float32)
        INPUT_GRID.append(batch_patched_grid)

    # offset convolution 관련
    zero_initializer = tf.zeros_initializer()
    offset = tf.layers.conv2d(input,
                              filters = 2,
                              kernel_size = (kernel_size, kernel_size),
                              strides = (1,1),
                              padding = 'same',
                              kernel_initializer = zero_initializer,
                              bias_initializer = zero_initializer,
                              name = 'offset_{}'.format(name))
    offset = tf.reshape(offset, [batch_size, input_size, input_size, -1, 2])
    off_x, off_y = offset[...,0], offset[...,1]

    OFFSET = []
    for offset in [off_x, off_y]:
        patched_offset = tf.image.extract_image_patches(offset,
                                                        ksizes = [1, kernel_size, kernel_size, 1],
                                                        strides = [1, 1, 1, 1],
                                                        rates = [1, 1, 1, 1],
                                                        padding = 'SAME')
        OFFSET.append(patched_offset)

    # 쌍선형 보간법에 대한 코드 시작
    x = tf.clip_by_value(INPUT_GRID[0] + OFFSET[0], 0, input_size - 1)
    y = tf.clip_by_value(INPUT_GRID[1] + OFFSET[1], 0, input_size - 1)
    x0, y0 = tf.cast(x, 'int32'), tf.cast(y, 'int32')
    x1, y1 = x0 + 1, y0 + 1
    x0, x1 = [tf.clip_by_value(i, 0, input_size - 1) for i in [x0, x1]]
    y0, y1 = [tf.clip_by_value(i, 0, input_size - 1) for i in [y0, y1]]
    indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]
    
    # input image에서 픽셀값 추출
    P = []
    for index in indices:
        tmp_y, tmp_x = index
        batch, h, w, n = tmp_y.get_shape().as_list()
        batch_idx = tf.reshape(tf.range(batch), (batch, 1, 1, 1))
        b = tf.tile(batch_idx, (1, h, w, n))
        pixel_idx = tf.stack([b, tmp_y, tmp_x], axis = -1)
        p = tf.gather_nd(input, pixel_idx) # input에서 픽셀값을 뽑아오는 것이라는 점에 주목해야함 
        P.append(p)

    # 쌍선형 보간법에 관한 내용
    # https://ballentain.tistory.com/55?category=778103
    x0, x1, y0, y1 = [tf.to_float(i) for i in [x0, x1, y0, y1]]
    w0 = (y1 - y) * (x1 - x)
    w1 = (y1 - y) * (x - x0)
    w2 = (y - y0) * (x1 - x)
    w3 = (y - y0) * (x - x0)
    w0, w1, w2, w3 = [tf.expand_dims(i, axis=-1) for i in [w0, w1, w2, w3]]
    pixels = tf.add_n([w0 * P[0], w1 * P[1], w2 * P[2], w3 * P[3]]) # 맨 뒤 3은 channel 뜻 함

    ## 어쨌거나 여태까지 처리한 좌표계에 대한 정보를 갖고
    ## 새로운 형태의 pixel 값을 갖는 행렬 생성함
    pixels = tf.reshape(pixels, [batch_size, input_size * 3, input_size * 3, -1])

    kernel_initializer = tf.contrib.layers.xavier_initializer()
    output_logits = tf.layers.conv2d(pixels,
                                     filters,
                                     kernel_size = (3,3),
                                     strides = (3,3), # strides로 중첩되는 걸 해결해줬네
                                     name = 'Logits_{}' .format(name),
                                     kernel_initializer = kernel_initializer) 
    return output_logits


if __name__ == '__main__':
    input = tf.constant(1., shape = (10, 15, 15, 3))
    dcl = deformable_conv(input, '1', batch_size = 10, filters = 32, kernel_size = 3)
