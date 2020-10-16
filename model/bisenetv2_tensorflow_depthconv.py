import tensorflow as tf


class ConvBNReLU(tf.keras.layers.Layer):
    def __init__(
        self,
        out_chan,
        ks=3,
        stride=1,
        padding="same",
        dilation=1,
        groups=1,
        bias=False,
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(
            out_chan,
            kernel_size=ks,
            strides=stride,
            padding=padding,
            dilation_rate=dilation,
            # groups=groups,
            use_bias=bias,
            kernel_regularizer=kernel_regularizer,
        )
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x, training=None):
        feat = self.conv(x)
        feat = self.bn(feat, training=training)
        feat = self.relu(feat)
        return feat


class DetailBranch(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.S1 = tf.keras.Sequential(
            [ConvBNReLU(64, 3, stride=2), ConvBNReLU(64, 3, stride=1)]
        )
        self.S2 = tf.keras.Sequential(
            [
                ConvBNReLU(64, 3, stride=2),
                ConvBNReLU(64, 3, stride=1),
                ConvBNReLU(64, 3, stride=1),
            ]
        )
        self.S3 = tf.keras.Sequential(
            [
                ConvBNReLU(128, 3, stride=2),
                ConvBNReLU(128, 3, stride=1),
                ConvBNReLU(128, 3, stride=1),
            ]
        )

    def call(self, x, training=None):
        feat = self.S1(x, training=training)
        feat = self.S2(feat, training=training)
        feat = self.S3(feat, training=training)
        return feat


class StemBlock(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv = ConvBNReLU(16, 3, stride=2)
        self.left = tf.keras.Sequential(
            [ConvBNReLU(8, 1, stride=1, padding="valid"), ConvBNReLU(16, 3, stride=2),]
        )
        self.right = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")
        self.fuse = ConvBNReLU(16, 3, stride=1)

    def call(self, x, training=None):
        feat = self.conv(x, training=training)
        feat_left = self.left(feat, training=training)
        feat_right = self.right(feat, training=training)
        feat = tf.keras.layers.concatenate([feat_left, feat_right])
        feat = self.fuse(feat, training=training)
        return feat


class CEBlock(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.bn = tf.keras.layers.BatchNormalization()
        self.conv_gap = ConvBNReLU(128, 1, stride=1, padding="valid")
        # TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvBNReLU(128, 3, stride=1)

    def call(self, x, training=None):
        # feat = self.global_avg_pool(x)
        feat = tf.reduce_mean(x, [1, 2], keepdims=True)
        feat = self.bn(feat, training=training)
        feat = self.conv_gap(feat, training=training)
        feat = feat + x
        feat = self.conv_last(feat, training=training)
        return feat


class GELayerS1(tf.keras.layers.Layer):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super().__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, 3, stride=1)
        self.dwconv = tf.keras.Sequential(
            [
                # tf.keras.layers.Conv2D(
                #     mid_chan,
                #     kernel_size=3,
                #     strides=1,
                #     padding="same",
                #     groups=in_chan,
                #     use_bias=False,
                # ),
                tf.keras.layers.DepthwiseConv2D(
                    depth_multiplier=exp_ratio,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    use_bias=False,
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),  # not shown in paper
            ]
        )
        self.conv2 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    out_chan, kernel_size=1, strides=1, padding="valid", use_bias=False
                ),
                tf.keras.layers.BatchNormalization(),
            ]
        )
        # self.conv2[1].last_bn = True WTF???
        self.relu = tf.keras.layers.ReLU()

    def call(self, x, training=None):
        feat = self.conv1(x, training=training)
        feat = self.dwconv(feat, training=training)
        feat = self.conv2(feat, training=training)
        feat = feat + x
        feat = self.relu(feat)
        return feat


class GELayerS2(tf.keras.layers.Layer):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super().__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, 3, stride=1)
        self.dwconv1 = tf.keras.Sequential(
            [
                # tf.keras.layers.Conv2D(
                #     mid_chan,
                #     kernel_size=3,
                #     strides=2,
                #     padding="same",
                #     groups=in_chan,
                #     use_bias=False,
                # ),
                tf.keras.layers.DepthwiseConv2D(
                    depth_multiplier=exp_ratio,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    use_bias=False,
                ),
                tf.keras.layers.BatchNormalization(),
            ]
        )
        self.dwconv2 = tf.keras.Sequential(
            [
                # tf.keras.layers.Conv2D(
                #     mid_chan,
                #     kernel_size=3,
                #     strides=1,
                #     padding="same",
                #     groups=mid_chan,
                #     use_bias=False,
                # ),
                tf.keras.layers.DepthwiseConv2D(
                    depth_multiplier=1,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    use_bias=False,
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),  # not shown in paper
            ]
        )
        self.conv2 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    out_chan, kernel_size=1, strides=1, padding="valid", use_bias=False
                ),
                tf.keras.layers.BatchNormalization(),
            ]
        )
        # self.conv2[1].last_bn = True WTF?
        self.shortcut = tf.keras.Sequential(
            [
                # tf.keras.layers.Conv2D(
                #     in_chan,
                #     kernel_size=3,
                #     strides=2,
                #     padding="same",
                #     groups=in_chan,
                #     use_bias=False,
                # ),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3, strides=2, padding="same", use_bias=False,
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    out_chan, kernel_size=1, strides=1, padding="valid", use_bias=False,
                ),
                tf.keras.layers.BatchNormalization(),
            ]
        )
        self.relu = tf.keras.layers.ReLU()

    def call(self, x, training=None):
        feat = self.conv1(x, training=training)
        feat = self.dwconv1(feat, training=training)
        feat = self.dwconv2(feat, training=training)
        feat = self.conv2(feat, training=training)
        shortcut = self.shortcut(x, training=training)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat


class SegmentBranch(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.S1S2 = StemBlock()
        self.S3 = tf.keras.Sequential([GELayerS2(16, 32), GELayerS1(32, 32)])
        self.S4 = tf.keras.Sequential([GELayerS2(32, 64), GELayerS1(64, 64)])
        self.S5_4 = tf.keras.Sequential(
            [
                GELayerS2(64, 128),
                GELayerS1(128, 128),
                GELayerS1(128, 128),
                GELayerS1(128, 128),
            ]
        )
        self.S5_5 = CEBlock()

    def call(self, x, training=None):
        feat2 = self.S1S2(x, training=training)
        feat3 = self.S3(feat2, training=training)
        feat4 = self.S4(feat3, training=training)
        feat5_4 = self.S5_4(feat4, training=training)
        feat5_5 = self.S5_5(feat5_4, training=training)
        return feat2, feat3, feat4, feat5_4, feat5_5


class BGALayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.left1 = tf.keras.Sequential(
            [
                # tf.keras.layers.Conv2D(
                #     128,
                #     kernel_size=3,
                #     strides=1,
                #     padding="same",
                #     groups=128,
                #     use_bias=False,
                # ),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3, strides=1, padding="same", use_bias=False,
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    128, kernel_size=1, strides=1, padding="valid", use_bias=False
                ),
            ]
        )
        self.left2 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    128, kernel_size=3, strides=2, padding="same", use_bias=False
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.AveragePooling2D(
                    pool_size=3, strides=2, padding="same"
                ),
            ]
        )
        self.right1 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    128, kernel_size=3, strides=1, padding="same", use_bias=False
                ),
                tf.keras.layers.BatchNormalization(),
            ]
        )
        self.right2 = tf.keras.Sequential(
            [
                # tf.keras.layers.Conv2D(
                #     128,
                #     kernel_size=3,
                #     strides=1,
                #     padding="same",
                #     groups=128,
                #     use_bias=False,
                # ),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3, strides=1, padding="same", use_bias=False,
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    128, kernel_size=1, strides=1, padding="valid", use_bias=False
                ),
            ]
        )
        # TODO: does this really has no relu?
        self.conv = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    128, kernel_size=3, strides=1, padding="same", use_bias=False
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),  # not shown in paper
            ]
        )

    def call(self, x_d, x_s, training=None):
        # dsize = x_d.size()[2:]
        dsize = tf.shape(x_d)[1:3]
        left1 = self.left1(x_d, training=training)
        left2 = self.left2(x_d, training=training)
        right1 = self.right1(x_s, training=training)
        right2 = self.right2(x_s, training=training)

        # right1 = F.interpolate(right1, size=dsize, mode="bilinear", align_corners=True)
        right1 = tf.compat.v1.image.resize(
            right1, size=dsize, align_corners=True
        )  # by default it is bilinear

        left = left1 * tf.keras.activations.sigmoid(right1)
        right = left2 * tf.keras.activations.sigmoid(right2)

        # right = F.interpolate(right, size=dsize, mode="bilinear", align_corners=True)
        right = tf.compat.v1.image.resize(
            right, size=dsize, align_corners=True
        )  # by default it is bilinear

        out = self.conv(left + right, training=training)
        return out


class SegmentHead(tf.keras.layers.Layer):
    def __init__(self, mid_chan, n_classes, output_size):
        super().__init__()
        self.output_size = output_size
        self.conv = ConvBNReLU(mid_chan, 3, stride=1)
        self.drop = tf.keras.layers.Dropout(0.1)
        self.conv_out = tf.keras.layers.Conv2D(
            n_classes, kernel_size=1, strides=1, padding="valid", use_bias=True
        )

    def call(self, x, training=None):
        feat = self.conv(x, training=training)
        feat = self.drop(feat, training=training)
        feat = self.conv_out(feat, training=training)
        feat = tf.compat.v1.image.resize(
            feat, size=self.output_size, align_corners=True
        )  # by default it is bilinear
        return feat


def get_bisenetv2(input_shape, n_classes):
    output_shape = input_shape[:2]
    detail = DetailBranch()
    segment = SegmentBranch()
    bga = BGALayer()

    # TODO: what is the number of main head channels ?
    head = SegmentHead(1024, n_classes, output_shape)
    aux2 = SegmentHead(128, n_classes, output_shape)
    aux3 = SegmentHead(128, n_classes, output_shape)
    aux4 = SegmentHead(128, n_classes, output_shape)
    aux5_4 = SegmentHead(128, n_classes, output_shape)

    x = tf.keras.Input(shape=input_shape, name="images")
    feat_d = detail(x)
    feat2, feat3, feat4, feat5_4, feat_s = segment(x)
    feat_head = bga(feat_d, feat_s)

    logits = head(feat_head)

    logits_aux2 = aux2(feat2)
    logits_aux3 = aux3(feat3)
    logits_aux4 = aux4(feat4)
    logits_aux5_4 = aux5_4(feat5_4)
    return tf.keras.Model(
        inputs=x,
        outputs=[logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4],
        name="BiSeNetV2",
    )


if __name__ == "__main__":
    import numpy as np
    import time

    input_shape = (640, 360, 3)
    model = get_bisenetv2(input_shape, n_classes=2)

    model.summary()
    # exit()
    # model.save("tfbisenet")
    model.compile("adam", "mse")
    model = tf.function(model)
    # tf.keras.utils.plot_model(model)

    image = tf.random.normal((1, *input_shape))
    # warm up
    for i in range(10):
        model(image)
    init = time.time()
    iters = 200
    for i in range(iters):
        model(image)
    end = time.time() - init
    print(f"FPS {1/(end/iters)}")
    print(f"Time {end/iters}")
