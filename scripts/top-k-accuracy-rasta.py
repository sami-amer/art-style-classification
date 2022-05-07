import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

# MUST BE IN RASTA FOLDER
os.chdir("rasta/")
tf.keras.backend.clear_session()

print(tf.version.VERSION)
print(tf.test.is_gpu_available())


def k_accuracy(output, target, topk=(1,)):
    
    maxk = max(topk)  # max number labels we will consider in the right choices for out model
    batch_size = target.shape[0]

    _,y_pred = tf.math.top_k(output,k=maxk)
    y_pred = tf.transpose(y_pred)  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.
    # print(y_pred)
    # target_reshaped = tf.repeat(tf.math.argmax(target.reshape(1, -1)),y_pred.shape[0])  # [B] -> [B, 1] -> [maxk, B]
    _, target_reshaped = tf.split(tf.where(target),2,axis=1)
    target_reshaped = tf.reshape(tf.repeat(target_reshaped,y_pred.shape[0]),(y_pred.shape[0],1))
    # _,target_reshaped = tf.transpose(tf.repeat(tf.split(tf.where(target),2,axis=1),y_pred.shape[0]))

    correct = (y_pred == tf.cast(target_reshaped,tf.int32))  # [maxk, B] were for each example we know which topk prediction matched truth
    # print(correct)
    # -- get topk accuracy
    list_topk_accs = []  # idx is topk1, topk2, ... etc
    for k in topk:
        # get tensor of which topk answer was right
        ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
        # print(ind_which_topk_matched_truth)
        # flatten it to help compute if we got it correct for each example in batch
        flattened_indicator_which_topk_matched_truth = tf.cast(tf.reshape(ind_which_topk_matched_truth,-1),tf.float32)  # [k, B] -> [kB]
        # get if we got it right for any of our top k prediction for each example in batch
        tot_correct_topk = tf.math.reduce_sum(tf.cast(flattened_indicator_which_topk_matched_truth,tf.float32),axis=0,keepdims=True)  # [kB] -> [1]
        # print(tot_correct_topk)
        # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
        topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
        list_topk_accs.append(topk_acc)
    # print(list_topk_accs)
    return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]



#rasta_model = tf.keras.models.load_model("models/default/model.h5")
#rasta_model.compile(
#    optimizer="rmsprop",
#    loss="categorical_crossentropy",
#    metrics=["categorical_accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5)],
#)

#rasta_model = tf.keras.models.load_model("models/default/model.h5")
rasta_model = tf.keras.models.load_model("../output_models/rasta_trained_extended_v2")

batch_size = 1
img_height = 224
img_width = 224

d_size = "extended"  # or full

test_dir = f"data/wikipaintings_{d_size}/wikipaintings_test"

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical",
)

classes= test_ds.class_names

test_ds = test_ds.map(
    lambda image, label: (
        tf.keras.applications.imagenet_utils.preprocess_input(image),
        label,
    )
)



# for image,labels in test_ds.unbatch().as_numpy_iterator():
#     image = np.expand_dims(image,axis=0)
#     output = rasta_model(image)
#     print(output)


correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
top_1_acc = 0
top_3_acc = 0
top_5_acc = 0
batch_count = 0

# again no gradients needed

for data in test_ds:
    images, labels = data
    outputs = rasta_model(images)
    batch_count+=1
    for output, label in zip(outputs,labels):
        # print(output)
        # print(label)
        top_k_accs = k_accuracy(outputs,labels,topk=(1,3,5))
        top_1_acc += top_k_accs[0].numpy()[0]
        top_3_acc += top_k_accs[1].numpy()[0]
        top_5_acc += top_k_accs[2].numpy()[0]
        # print(output)
        prediction = tf.math.argmax(output)
        label=tf.math.argmax(label)
        # print(prediction)
        # print(label)
    # collect the correct predictions for each class
        if label == prediction:
            correct_pred[classes[label]] += 1
        total_pred[classes[label]] += 1

    # for batch in dataloaders["test"]:
            # image,label = batch
            # output=model(image)
            # # print(output)


total_correct = 0
pred_tot = 0
# print accuracy for each class
for classname, correct_count in correct_pred.items():
    total_correct += correct_count
    pred_tot += total_pred[classname]
    try:
        accuracy = 100 * float(correct_count) / total_pred[classname]
    except:
        print(classname)
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

total_acc = 100 * total_correct/pred_tot
k_1_acc =  100 * top_1_acc / batch_count
k_3_acc =  100 * top_3_acc / batch_count
k_5_acc =  100 * top_5_acc / batch_count

print(f'Accuracy is {total_acc:.1f} %')
print(f'Accuracy for top 1: is {k_1_acc:.1f} %')
print(f'Accuracy for top 3: is {k_3_acc:.1f} %')
print(f'Accuracy for top 5: is {k_5_acc:.1f} %')