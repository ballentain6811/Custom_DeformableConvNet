import numpy as np
import tensorflow as tf

class DeformableConv2D(tf.keras.layers.Conv2D):
    def __init__(self,
                 batch_size,
                 filters,
                 kernel_size,
                 name,
                 kernel_initializer,
                 **kwargs):
        
        self.batch_size = batch_size       
        super().__init__(
            filters = filters,
            kernel_size = kernel_size,
            name = name,
            kernel_initializer = kernel_initializer,
            **kwargs)
        
        self.kernel = None
        self.bias = None
        self.offset_kernel = None
        self.offset_bias = None
        
    
    def build(self, input_shape):
        input_dim = int(input_shape[-1])
            
        # conv2d
        self.kernel = self.add_weight(
            name = 'kernel_{}' .format(self.name),
            shape = self.kernel_size + (input_dim, self.filters),
            initializer = self.kernel_initializer,
            trainable=True
        )
        
        self.bias = self.add_weight(
            name = 'bias_{}' .format(self.name),
            shape = (self.filters,),
            initializer=self.kernel_initializer,
            trainable=True
        )
        
        
        # offset
        self.offset_kernel = self.add_weight(
            name = 'offset_kernel_{}'.format(self.name),
            shape = self.kernel_size + (input_dim, 2),
            initializer = tf.zeros_initializer(),
            trainable = True            
        )
        
        self.offset_bias = self.add_weight(
            name = 'offset_bias_{}' .format(self.name),
            shape = (2,),
            initializer = tf.zeros_initializer(),
            trainable = True
        )
                
        self.built = True # 또는 super(DeformableConv2D, self).build(input_shape) 인 듯

        
    def call(self, input):
        return self.deformable_conv(input, self.name, self.batch_size, self.filters, self.kernel_size)
        

    def deformable_conv(self, input, name, batch_size, filters, kernel_size):
        input_size = input.get_shape().as_list()[1]

        # Input image 좌표계 관련
        grid_x, grid_y = tf.meshgrid(tf.range(input_size), tf.range(input_size))
        INPUT_GRID = []
        for grid in [grid_x, grid_y]:
            grid = tf.reshape(grid, [1, *grid.get_shape(), 1])
            patched_grid = tf.image.extract_image_patches(grid,
                                                          ksizes = (1,) + kernel_size + (1,),
                                                          strides = [1, 1, 1, 1],
                                                          rates = [1, 1, 1, 1],
                                                          padding = 'SAME')
            batch_patched_grid = tf.tile(patched_grid, [batch_size, 1, 1, 1])
            batch_patched_grid = tf.cast(batch_patched_grid, tf.float32)
            INPUT_GRID.append(batch_patched_grid)

        # offset convolution 관련
        offset = tf.nn.conv2d(input,
                              filter = self.offset_kernel,
                              strides = [1, 1, 1, 1],
                              padding = 'SAME'
                              )
        offset += self.offset_bias
        offset = tf.reshape(offset, [batch_size, input_size, input_size, -1, 2])
        off_x, off_y = offset[...,0], offset[...,1]

        # 추가    
        OFFSET = []
        for offset in [off_x, off_y]:
            patched_offset = tf.image.extract_image_patches(offset,
                                                            ksizes = (1,) + kernel_size + (1,),
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

        # _get_pixel_values_at_point( ) 부분
        P = []
        for index in indices:
            tmp_y, tmp_x = index
            batch, h, w, n = tmp_y.get_shape().as_list()

            batch_idx = tf.reshape(tf.range(batch), (batch, 1, 1, 1))
            b = tf.tile(batch_idx, (1, h, w, n))
            pixel_idx = tf.stack([b, tmp_y, tmp_x], axis = -1)
            p = tf.gather_nd(input, pixel_idx)
            P.append(p)

        # 쌍선형 보간법에 관한 내용
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
        output_logits = tf.nn.conv2d(pixels,
                                     filter = self.kernel,
                                     strides = [1, 3, 3, 1],
                                     padding = 'VALID')
        output_logits += self.bias
        return output_logits
    
    def _inference_grid_offset(self, input_images):
        if len(input_images.shape) < 4:
            raise "No"
        b, h, w, c = input_images.shape
        input_tensor = tf.placeholder(tf.float32, [None, h, w, c])
        grid_offset = tf.nn.conv2d(input_tensor,
                                   filter = self.offset_kernel,
                                   strides = [1, 1, 1, 1],
                                   padding = 'SAME')
        grid_offset += self.offset_bias
        
        sess = tf.keras.backend.get_session()
        offset = sess.run(grid_offset, feed_dict = {input_tensor : input_images})
        return offset
        
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    kernel_initializer = tf.contrib.layers.xavier_initializer()
    dclk = DeformableConv2D(batch_size = 10,
                            filters = 32,
                            kernel_size = (3,3),
                            name = 'layer1',
                            kernel_initializer=kernel_initializer)
    






    























