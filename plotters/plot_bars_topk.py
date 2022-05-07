import matplotlib.pyplot as plt
import numpy as np


labels_pre = [
    'Abstract Art', 'Abstract Expressionism', 'Informel', 'Nouveau',
    'Baroque', 'Color Field Painting', "Cubism", "Early Renaissance",
    "Expressionism","High Rennaissance", "Impressionism","Magic Realism",
    "Mannerism","Minimalism","Primitivism","Neoclassicism","Northern Renaissance",
    "Pop Art","Post Impressionism","Realism","Rococo","Romanticism","Surrealism","Symbolism","Ukiyo-e"]

labels_extended = [
    'Abstract Art', 'Abstract Expressionism', 'Informel', 'Nouveau',
    'Baroque', "Chinese Landscapes",'Color Field Painting', "Cubism", "Early Renaissance",
    "Expressionism","High Rennaissance", "Impressionism","Islamic Art","Islamic Textiles","Magic Realism",
    "Mannerism","Minimalism","Primitivism","Neoclassicism","Northern Renaissance",
    "Pop Art","Post Impressionism","Realism","Rococo","Romanticism","Surrealism","Symbolism","Ukiyo-e (extended)"]

imageNet_pre_accs = [
    31.0,36.8,8.3,36.8,41.4,59.2,35.1,49.6,32.9,
    17.3,68.3,26.7,21.8,70.6,26.7,47.0,39.2,33.9,
    24.2,58.4,30.2,41.8,50.5,25.1,65.8]
imageNet_pre_topk =[44.1,72.5,84.1]

imageNet_ext_accs = [
    26.0,37.3,10.4,36.8,46.4,89.1,58.4,39.3,
    48.9,31.5,20.5,70.2,78.3,88.7,26.7,20.2,69.0,
    21.8,45.2,41.7,33,22.5,54.8,31.8,39.9,44.3,26.6,88.1]
imageNet_ext_topk = [47.5,73.8,85.0]

styleNet_pre_accs=[
    26.0,37.8,11.5,41.1,45.1,59.2,43.5,
    45.9,36.0,16.5,70.2,24.8,18.5,64.3,33.2,54.1,
    38.3,37.5,26.5,49.3,28.6,38.0,46.1,28.1,65.0]
styleNet_pre_topk =[43.9,72.2,83.7]

styleNet_ext_accs=[
    27.0,38.8,10.4,35.1,46.1,88.2,59.2,39.3,51.9,36.5,
    15.0,70.1,80.4,91.2,24.8,19.3,61.1,29.2,50.0,32.9,
    34.8,24.9,52.3,28.6,36.7,41.8,27.1,87.8]
styleNet_ext_topk =[47.4,74.0,84.7]

rasta_pre_accs=[
    47.0,40.7,25.0,63.5,70.9,72.8,59.5,62.4,45.5,
    48.8,68.5,49.5,38.7,68.3,58.4,64.4,80.8,63.4,
    47.7,64.1,64.6,64.8,66.6,43.1,74.4]
rasta_pre_topk=[59.2,84.2,92.6]

rasta_ext_accs=[
    41.0,50.2,43.8,57.7,75.7,98.6,77.6,
    61.3,72.2,59.1,57.5,73.8,88.0,92.5,
    50.5,35.3,71.4,60.9,62.2,78.3,53.6,
    42.1,64.1,64.6,59.6,59.5,52.7,94.6]
rasta_ext_topk=[64.9,87.6,94.4]
topk_labels = ["Top-1","Top-3","Top-5"]

plt.rcParams['figure.dpi'] = 140

# ------ ImageNet and StyleNet and RASTA

x = np.arange(len(topk_labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, imageNet_pre_topk, width, label='ImageNet',)
rects2 = ax.bar(x , styleNet_pre_topk, width, label='StyleNet')
rects3 = ax.bar(x + width, rasta_pre_topk, width, label='RASTA')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Top-K Accuracy')
ax.set_title('Accuracy per Model')
ax.set_xticks(x, topk_labels)

# plt.axvline(x=59.2,color="black") # Rasta Line

# for tick in ax.xaxis.get_major_ticks()[1::2]:
    # tick.set_pad(5)
ax.legend()
# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.savefig("plots/bars_topk/TopK bar.png")

plt.clf()

# ------ ImageNet and StyleNet and RASTA EXTENDED

x = np.arange(len(topk_labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, imageNet_ext_topk, width, label='ImageNet',)
rects2 = ax.bar(x , styleNet_ext_topk, width, label='StyleNet')
rects3 = ax.bar(x + width, rasta_ext_topk, width, label='RASTA')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Top-K Accuracy')
ax.set_title('Accuracy per Model Ext.')
ax.set_xticks(x, topk_labels)

# plt.axvline(x=59.2,color="black") # Rasta Line

# for tick in ax.xaxis.get_major_ticks()[1::2]:
    # tick.set_pad(5)
ax.legend()
# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.savefig("plots/bars_topk/TopK bar ext.png")

plt.clf()