from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2

# Made using:
# https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/


class GradCAM:
    def __init__(self, model, classIndex=None, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map`
        self.model = model
        self.classIndex = classIndex
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # Find the last convolutional layer by looping trough
        # the layers in reverse and returning the name of the first one with 4 dimensions
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        # If there are no 4D Layers GradCAM cannot be applied.
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM")

    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying:
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
            self.model.output]
        )
        # Record operation to use during differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)

            if self.classIndex is not None:
                loss = predictions[:, self.classIdx]
            else:
                loss = predictions[0][0]

        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # Compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, 'float32')
        castGrads = tf.cast(grads > 0, 'float32')

        # the values are all either 0 or 1.
        # By multiplying we only keep the positive values.
        guidedGrads = castConvOutputs * castGrads * grads

        # The convolution and guided have an unneeded batch dimension
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # Get the dimensions of the input image
        (w, h) = (image.shape[2], image.shape[1])
        # Resize the class activation map (cam) to match the input
        heatmap = cv2.resize(cam.numpy(), (w, h))

        # Normalize the heatmap to values between 0 - 1
        numerator = heatmap - np.min(heatmap)
        denominator = (heatmap.max() - heatmap.min()) + eps
        heatmap = numerator / denominator

        # Scale values to be between 0 - 255 & convert floats to unsigned 8-bit int
        heatmap = (heatmap * 255).astype('uint8')

        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS):
        # Apply the supplied color map to the heatmap & overlay this on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        return (heatmap, output)
