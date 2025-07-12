import os
import warnings
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from glob import glob
from itertools import chain
from sklearn.model_selection import train_test_split
# 设置绘图风格
sns.set_style('whitegrid')
# 忽略警告
warnings.filterwarnings('ignore')
# 打印可用GPU和使用的模型
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



# ======================================================================Part 1 command-line parameter======================================================================
# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, 
                    choices=[
                        'vgg16', 'densenet121', 'resnet101', 'mobilenetV3Large', 'inceptionV3',
                        'vit_b16', 'vit_b32', 'vit_l16', 'vit_l32',
                        'convnext_small', 'convnext_base', 'convnext_large'],
                    help="Choose a model from: vgg16, densenet121, resnet101, mobilenetV3Large, inceptionV3, "
                         "vit_b16, vit_b32, vit_l16, vit_l32, convnext_small, convnext_base, convnext_large.")
parser.add_argument('--learning_rate', type=float, help='Learning rate for model training')
parser.add_argument('--batch_size', type=int, help='Batch size for model training')
parser.add_argument('--epochs', type=int, help='Number of epochs for model training')
parser.add_argument('--stop_patience', type=int, help='Early stopping patience. Number of epochs with no improvement before stopping training.')
args = parser.parse_args()

# 确保用户指定了所有必要的参数
missing_args = []
if not args.model:
    missing_args.append("--model")
if args.learning_rate is None:
    missing_args.append("--learning_rate")
if args.batch_size is None:
    missing_args.append("--batch_size")
if args.epochs is None:
    missing_args.append("--epochs")
if args.stop_patience is None:
    missing_args.append("--stop_patience")
if missing_args:
    parser.error(f"The following arguments are required: {', '.join(missing_args)}")

# 打印model epoch batch size learning rate
print(f"Using model: {args.model}")
print(f"Learning Rate: {args.learning_rate}")
print(f"Batch Size: {args.batch_size}")
print(f"Epochs: {args.epochs}")
print(f"Stop Patience: {args.stop_patience}")



# ======================================================================Part 2: The data Preprocess======================================================================
# 读取 CSV 文件，确保路径正确
data = pd.read_csv('/home2/grkp39/My_project/Datasets/Data_Entry_2017.csv')   # 替换为实际路径

# 去除年龄大于100的异常数据并且将年龄数据转换为整数
data = data[data['Patient Age'] < 100]
data['Patient Age'] = data['Patient Age'].map(lambda x: int(x))

# 获取图像路径，并存入字典
data_image_paths = {
    os.path.basename(x): x for x in glob(os.path.join('/home2/grkp39/My_project/Datasets', 'images*', '*', '*.png'))  # 替换为实际路径
}

# 将图像路径添加到数据框中同时确保路径为字符串并删除缺失路径的行
data['path'] = data['Image Index'].map(data_image_paths.get)
data['path'] = data['path'].astype(str)
data = data.dropna(subset=['path'])

# 打印数据概况，随机查看3条样本数据
print('Scans found:', len(data_image_paths), ', Total Headers:', data.shape[0])
print(data.sample(3))

# 提取所有独特的标签并且打印全部标签
all_labels = np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x) > 0]
print('All Labels ({}): {}'.format(len(all_labels), all_labels))

# 为每个病症标签创建独立的列
for c_label in all_labels:
    if len(c_label) > 1:  # 移除空标签
        data[c_label] = data['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

# 首先创建所有样本的 'disease_vec' 列
data['disease_vec'] = data[all_labels].values.tolist()

# 加载图像索引
test_list_path = '/home2/grkp39/My_project/Datasets/test_list.txt'  # 替换为实际路径
if not os.path.exists(test_list_path):
    raise FileNotFoundError(f"Test list file not found at {test_list_path}")
with open(test_list_path, 'r') as f:
    test_image_names = f.read().splitlines()

# 分离训练数据和测试数据
train_data = data[~data['Image Index'].isin(test_image_names)]
test_data = data[data['Image Index'].isin(test_image_names)].dropna(subset=['path'])

# 从训练数据中分离验证集
train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=42)
print(f"Data size for training: {len(train_data)}")
print(f"Data size for validation: {len(valid_data)}")
print(f"Data size for testing: {len(test_data)}")

# 创建可视化保存目录
vis_dir = './visualizations_AG'
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir, exist_ok=True)

# 数据分布可视化
plt.rcParams.update({
    "font.family": "Times New Roman",   # 或者 'serif'
    "font.size": 12                     # 正文大小是10pt
})
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
data[all_labels].sum().sort_values(ascending=False).plot(kind='bar')
plt.title('Disease Distribution in Dataset')
plt.xlabel('Disease')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.subplot(1, 2, 2)
sns.histplot(data=data, x='Patient Age', bins=30)
plt.title('Age Distribution in Dataset')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'data_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()



# ============================================================Part 3 Data Load and preprocessing with different models============================================================
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.densenet import preprocess_input as densenet_preprocess_input
from keras.applications.resnet import preprocess_input as resnet_preprocess_input
from keras.applications.mobilenet_v3 import preprocess_input as mobilenet_v3_preprocess_input
from keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from keras.applications.convnext import preprocess_input as convnext_preprocess_input
# 设置路径和保存计数器
aug_vis_dir = './augmented_samples'
os.makedirs(aug_vis_dir, exist_ok=True)
_saved_image_paths = []


def vit_preprocess_input(image):
    return image / 255.0

def save_aug_image_py(img, filepath, suffix):
    if len(_saved_image_paths) >= 100:
        return
    filename = os.path.basename(filepath.numpy().decode())
    save_path = os.path.join(aug_vis_dir, f"aug_{len(_saved_image_paths)}_{suffix}_{filename}")
    img_np = img.numpy()
    img_pil = Image.fromarray(np.uint8(img_np))
    img_pil.save(save_path)
    _saved_image_paths.append(save_path)

def dataframe_to_dataset(df, model_name, img_size=(224, 224), training=True):
    paths = df['path'].values
    labels = np.array(df['disease_vec'].tolist(), dtype='float32')

    def load_and_preprocess_image(path, label):
        # 读取图像
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, img_size)
        orig = tf.identity(image)

        if training:
            # 原始增强流程
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            image = tf.image.random_crop(image, [img_size[0] - 10, img_size[1] - 10, 3])
            image = tf.image.resize(image, img_size)

            # 分别保存四种确定性增强（前 100 张，每种一张）
            if len(_saved_image_paths) < 100:
                flip = tf.image.flip_left_right(orig)
                tf.py_function(save_aug_image_py, [flip, path, 'flip'], [])

                bright = tf.image.adjust_brightness(orig, delta=0.1)
                tf.py_function(save_aug_image_py, [bright, path, 'brightness'], [])

                contrast = tf.image.adjust_contrast(orig, contrast_factor=1.2)
                tf.py_function(save_aug_image_py, [contrast, path, 'contrast'], [])

                cropped = tf.image.crop_to_bounding_box(orig, 5, 5, img_size[0] - 10, img_size[1] - 10)
                cropped = tf.image.resize(cropped, img_size)
                tf.py_function(save_aug_image_py, [cropped, path, 'crop'], [])

        # 模型特定预处理
        if model_name == 'vgg16':
            image = vgg16_preprocess_input(image)
        elif model_name == 'densenet121':
            image = densenet_preprocess_input(image)
        elif model_name == 'resnet101':
            image = resnet_preprocess_input(image)
        elif model_name == 'mobilenetV3Large':
            image = mobilenet_v3_preprocess_input(image)
        elif model_name == 'inceptionV3':
            image = inception_preprocess_input(image)
        elif model_name in ['vit_b16', 'vit_b32', 'vit_l16', 'vit_l32']:
            image = vit_preprocess_input(image)
        elif model_name in ['convnext_small', 'convnext_base', 'convnext_large']:
            image = convnext_preprocess_input(image)
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset



# ======================================================================Part 4 Models======================================================================
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications import InceptionV3
from vit_keras import vit
from tensorflow.keras.applications.convnext import ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge
from keras.layers import Input, Conv2D, Add, Activation, Multiply, BatchNormalization, GlobalAveragePooling2D, Concatenate, Dropout, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model

# Attention Gate 模块
def attention_gate(input_tensor, gating_tensor, inter_channels):
    """优化后的 Attention Gate 机制"""
    # Step 1: 1x1 Conv 降维
    theta_x = Conv2D(inter_channels, (1, 1), strides=(1, 1), padding='same')(input_tensor)
    phi_g = Conv2D(inter_channels, (1, 1), strides=(1, 1), padding='same')(gating_tensor)
    # Step 2: 结合特征 + ReLU
    add_xg = Add()([theta_x, phi_g])
    act_xg = Activation('relu')(add_xg)
    # Step 3: 生成权重掩码（改为 softmax 归一化）
    psi = Conv2D(1, (1, 1), strides=(1, 1), activation='sigmoid', padding='same')(act_xg)
    # Step 4: 应用权重掩码
    output = Multiply()([input_tensor, psi])
    # Step 5: Skip Connection（保持信息）
    output = Add()([output, input_tensor])
    return output

def insert_attention_gate(model_name, base_model_output):
    if model_name == 'vgg16':
        x = base_model_output.get_layer('block5_conv3').output

    elif model_name == 'densenet121':
        x = base_model_output.get_layer('conv5_block16_concat').output

    elif model_name == 'resnet101':
        x = base_model_output.get_layer('conv5_block3_out').output

    elif model_name == 'mobilenetV3Large':
        x = base_model_output.get_layer('Conv_1').output

    elif model_name == 'inceptionV3':
        x = base_model_output.get_layer('mixed10').output

    elif model_name in ['convnext_small', 'convnext_base', 'convnext_large']:
        # 根据不同模型选最后一层 block
        if model_name == 'convnext_small':
            block_layer_name = 'convnext_small_stage_3_block_2_pointwise_conv_2'
        elif model_name == 'convnext_base':
            block_layer_name = 'convnext_base_stage_3_block_2_pointwise_conv_2'
        elif model_name == 'convnext_large':
            block_layer_name = 'convnext_large_stage_3_block_2_pointwise_conv_2'
        x = base_model.get_layer(block_layer_name).output

    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    # 添加 Attention Gate（只加一层）
    gating_tensor = BatchNormalization()(Conv2D(128, (1, 1), padding='same')(x))
    x = attention_gate(x, gating_tensor, inter_channels=64)
    x = GlobalAveragePooling2D()(x)
    return x

IMG_SIZE = (224, 224, 3)
img_in = Input(shape=IMG_SIZE)

# 构建模型
if args.model == 'vgg16':
    base_model = VGG16(
        include_top=False, 
        weights='imagenet', 
        input_tensor=img_in
        )
    x = insert_attention_gate('vgg16', base_model)
    # VGG16 额外全连接层
    print("Applying extra dense layers for VGG16")
    x = Flatten()(x) 
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)

elif args.model == 'densenet121':
    base_model = DenseNet121(
        include_top=False, 
        weights='imagenet', 
        input_tensor=img_in
    )
    x = insert_attention_gate('densenet121', base_model)
    x = Dropout(0.5)(x)

elif args.model == 'resnet101':
    base_model = ResNet101(
        include_top=False, 
        weights='imagenet', 
        input_tensor=img_in, 
    )
    x = insert_attention_gate('resnet101', base_model)
    x = Dropout(0.5)(x)

elif args.model == 'mobilenetV3Large':
    base_model = MobileNetV3Large(
        include_top=False, 
        weights='imagenet', 
        input_tensor=img_in
    )
    x = insert_attention_gate('mobilenetV3Large', base_model)
    x = Dropout(0.5)(x)

elif args.model == 'inceptionV3':
    base_model = InceptionV3(
        include_top=False, 
        weights='imagenet', 
        input_tensor=img_in
    )
    x = insert_attention_gate('inceptionV3', base_model)
    x = Dropout(0.5)(x)

elif args.model in ['vit_b16', 'vit_b32', 'vit_l16', 'vit_l32']:
    base_model = getattr(vit, args.model)(
        image_size=IMG_SIZE[0], 
        pretrained=True, 
        include_top=False, 
        pretrained_top=False
    )
    x = base_model(img_in)
    x = Dropout(0.5)(x)

elif args.model == 'convnext_small':
    base_model = ConvNeXtSmall(
        include_top=False, 
        weights='imagenet', 
        input_tensor=img_in, 
    )
    x = insert_attention_gate('convnext_small', base_model)
    x = Dropout(0.5)(x)

elif args.model == 'convnext_base':
    base_model = ConvNeXtBase(
        include_top=False, 
        weights='imagenet', 
        input_tensor=img_in, 
    )
    x = insert_attention_gate('convnext_base', base_model)
    x = Dropout(0.5)(x)

elif args.model == 'convnext_large':
    base_model = ConvNeXtLarge(
        include_top=False, 
        weights='imagenet', 
        input_tensor=img_in, 
    )
    x = insert_attention_gate('convnext_large', base_model)
    x = Dropout(0.5)(x)

else:
    raise ValueError("Invalid model name. Please check your model argument.")

# 最后的全连接层用于多标签分类任务
predictions = Dense(len(all_labels), activation="sigmoid", name="predictions")(x)
model = Model(inputs=img_in, outputs=predictions)

# 优化器
optimizer = Adam(learning_rate=args.learning_rate) # Adam优化器

# 编译模型
model.compile(
    optimizer=optimizer, 
    # loss="binary_crossentropy", # 二元交叉熵loss function
    loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0), # focal loss function
    metrics=[
        tf.keras.metrics.AUC(name='auc', multi_label=True),   # 确保使用多标签AUC
    ]
)
print("Model successfully compiled.")



# ======================================================================Part 5 Training======================================================================
# 创建训练、验证和测试集的数据集
batch_size = args.batch_size
# 训练数据（带数据增强）
train_ds = dataframe_to_dataset(train_data, model_name=args.model, training=True).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
# 验证数据（无数据增强）
valid_ds = dataframe_to_dataset(valid_data, model_name=args.model, training=False).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
# 测试数据（无数据增强）
test_ds = dataframe_to_dataset(test_data, model_name=args.model, training=False).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

# 计算 steps_per_epoch 和 validation_steps
steps_per_epoch = np.ceil(len(train_data) / batch_size).astype(int)
validation_steps = np.ceil(len(valid_data) / batch_size).astype(int)
test_steps = np.ceil(len(test_data) / batch_size).astype(int)

# 设置模型检查点，保存最优模型（完整模型）
save_dir = './models_AG'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
checkpoint_filepath = os.path.join(save_dir, f'best_model_{args.model}.h5')
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_auc', 
    mode='max',
    save_best_only=True,
    save_weights_only=True  # **只保存最优权重**
)

# 设置早停机制，避免过拟合
early_stopping_callback = EarlyStopping(
    monitor='val_auc',
    patience=args.stop_patience,
    restore_best_weights=True
)

# 训练模型
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=args.epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[model_checkpoint_callback, early_stopping_callback]
)

# 加载最优权重
model.load_weights(checkpoint_filepath)

# HDF5 文件路径
final_model_path_h5 = os.path.join(save_dir, f'final_model_{args.model}.h5')

# 保存完整模型
model.save(final_model_path_h5, save_format='h5', include_optimizer=True)
print(f"训练完成！最优模型已保存至：\n  {final_model_path_h5}")

# 训练过程可视化
fig, axes = plt.subplots(2, 1, figsize=(10, 8))  # 改为3行以容纳AUC图

# 绘制训练和验证损失
axes[0].plot(history.history['loss'], label='Training Loss', color='b', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', color='r', linewidth=2)
axes[0].set_xlabel('Epochs', fontsize=14)
axes[0].set_ylabel('Loss', fontsize=14)
axes[0].set_title('Training and Validation Loss', fontsize=16)
axes[0].legend(fontsize=12)
axes[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# 绘制训练和验证AUC
axes[1].plot(history.history['auc'], label='Training AUC', color='b', linewidth=2)
axes[1].plot(history.history['val_auc'], label='Validation AUC', color='r', linewidth=2)
axes[1].set_xlabel('Epochs', fontsize=14)
axes[1].set_ylabel('AUC', fontsize=14)
axes[1].set_title('Training and Validation AUC', fontsize=16)
axes[1].legend(fontsize=12)
axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

plt.tight_layout(pad=1.0)
plt.savefig(os.path.join(vis_dir, f'training_history_{args.model}.png'), dpi=300, bbox_inches='tight')
plt.show()


# ======================================================================Part 6 Testing======================================================================
from keras.models import load_model
from sklearn.metrics import roc_curve
# 注册 Focal Loss
custom_objects = {
    "SigmoidFocalCrossEntropy": tfa.losses.SigmoidFocalCrossEntropy
}

# === 如果是 ConvNeXt，就先定义 LayerScale 类以防报错 ===
if "convnext" in args.model.lower():
    class LayerScale(tf.keras.layers.Layer):
        def __init__(self, init_values=1e-6, projection_dim=None, **kwargs):
            super().__init__(**kwargs)
            self.init_values = init_values
            self.projection_dim = projection_dim

        def build(self, input_shape):
            dim = self.projection_dim or input_shape[-1]
            self.gamma = self.add_weight(
                name="gamma",
                shape=(dim,),
                initializer=tf.keras.initializers.Constant(self.init_values),
                trainable=True,
            )

        def call(self, x):
            return x * self.gamma

    # 先注册进去用于检测
    custom_objects["LayerScale"] = LayerScale

    # 检查模型是否真的用到了 LayerScale（可选调试）
    temp_model = load_model(final_model_path_h5, custom_objects=custom_objects, compile=False)
    has_layer_scale = any('layer_scale' in layer.name for layer in temp_model.layers)
    if has_layer_scale:
        print("Detected LayerScale in the model.")
    else:
        print("No LayerScale detected, but custom_objects still safe to use.")

# === 正式加载模型（带上 custom_objects） ===
model = load_model(final_model_path_h5, custom_objects=custom_objects)

# 测试
test_results = model.evaluate(test_ds, steps=test_steps)
print(f'Test Loss: {test_results[0]:.4f}')

# 预测并绘制每个类别的ROC曲线
predictions = model.predict(test_ds, steps=test_steps)
plt.figure(figsize=(12, 8))
aucs = []  # 存储每个类别的 AUC 值

print("\n=== 每个疾病的 AUC 和最佳阈值 ===")
for i, label in enumerate(all_labels):
    true_labels = test_data[label].values
    pred_probs = predictions[:, i]
    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
    # 计算 AUC
    auc_metric = tf.keras.metrics.AUC()
    auc_metric.update_state(true_labels, pred_probs)
    roc_auc = auc_metric.result().numpy()
    aucs.append(roc_auc)
    # 计算最佳阈值（Youden's J statistic）
    youden_index = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_index)]
    # 打印 AUC 和最佳阈值
    print(f"{label}: AUC = {roc_auc:.3f}, Best Threshold = {best_threshold:.3f}")
    # 绘制 ROC 曲线
    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC: {roc_auc:.2f})')

# 计算并打印平均 AUC
mean_auc = np.mean(aucs)
print(f'\nMean AUC across all classes: {mean_auc:.4f}')

# 绘制 ROC 曲线
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curves for Each Disease Label (Mean AUC: {mean_auc:.2f})')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=10, ncol=3)
plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig(os.path.join(vis_dir, f'roc_curves_{args.model}.png'), dpi=300, bbox_inches='tight')
plt.show()

