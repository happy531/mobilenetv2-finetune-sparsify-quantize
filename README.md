# Plant Disease Classification on Mobile Devices

## Project Overview

This project aims to build a **compact and efficient deep learning model** for recognizing **plant diseases**, suitable for deployment on **mobile devices**. It is designed to assist farmers and growers in remote areas without internet access in quickly diagnosing plant diseases using their smartphones.

---

## Dataset and Base Model

* **Dataset:** [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) from Kaggle

  * 16 classes of plant diseases
  * Over 20,000 labeled JPG images

* **Base Model:** `MobileNetV2`

  * Pre-trained on ImageNet
  * Lightweight: 9.35 MB with 3.4M parameters

---

## Classes

```text
Pepper__bell___Bacterial_spot
Pepper__bell___healthy
Potato___Early_blight
Potato___Late_blight
Potato___healthy
Tomato_Bacterial_spot
Tomato_Early_blight
Tomato_Late_blight
Tomato_Leaf_Mold
Tomato_Septoria_leaf_spot
Tomato_Spider_mites_Two_spotted_spider_mite
Tomato__Target_Spot
Tomato__Tomato_YellowLeaf__Curl_Virus
Tomato__Tomato_mosaic_virus
Tomato__healthy
```

---

## Data Preprocessing

* **Normalization:** Images scaled to range `[0, 1]`
* **Augmentation Techniques:**

  * Rotation
  * Shifting
  * Shearing
  * Zooming
  * Horizontal flipping
* **Validation Split:** Ensures generalization and avoids overfitting

---

## Model Architecture & Training

### Transfer Learning with MobileNetV2

```python
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Training Parameters

* **Epochs:** 10
* **Optimizer:** Adam
* **Loss:** Categorical Crossentropy
* **Training Accuracy:** \~90% (on epoch 8, after that it started to overfit.)
* **Validation Accuracy:** \~88%
* **Model starts overfitting after epoch 8**

---

## Model Simplification with Sparsification

A custom Keras callback was used to **zero-out weights** with absolute values below a defined threshold after each training batch:

```python
class SparsifyCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def on_train_batch_end(self, batch, logs=None):
        for layer in self.model.layers:
            if hasattr(layer, 'kernel'):
                weights = layer.kernel
                mask = tf.math.abs(weights) >= self.threshold
                layer.kernel.assign(weights * tf.cast(mask, weights.dtype))
```

### Benefits:

* Reduces model size
* Improves inference speed on edge devices
* Adds regularization effect to reduce overfitting

---

## Post-Training Quantization

The model is quantized to TensorFlow Lite format (`INT8`) for mobile deployment.

```python
def representative_data_gen():
    for _ in range(100):
        image, _ = next(iter(train_generator))
        yield [image.astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_quant_model = converter.convert()

with open('mobilenetv2_int8.tflite', 'wb') as f:
    f.write(tflite_quant_model)
```

### Results:

* **Original size:** 9.35 MB
* **Quantized size:** 2.6 MB
* **Accuracy retained post-quantization**

---

## Model Optimization Techniques Summary

| Technique               | Purpose                                                |
| ----------------------- | ------------------------------------------------------ |
| **Data Augmentation**   | Improve generalization, simulate real-world conditions |
| **Dropout & BatchNorm** | Prevent overfitting and stabilize training             |
| **Transfer Learning**   | Leverage pre-trained knowledge for faster training     |
| **Sparsification**      | Reduce model complexity and size                       |
| **Quantization (INT8)** | Optimize for mobile/edge deployment                    |

---

## Final Outcome

A highly optimized, mobile-friendly plant disease classification model using MobileNetV2, ready for deployment in mobile or embedded applications.