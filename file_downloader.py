import requests
import json
import wget
import pandas as pd
import pickle

# df = pd.read_csv("MetObjects.csv")


# with open(r"MetObjects.pickle", "wb") as output_file:
#      pickle.dump(df, output_file)

with open(r"MetObjects.pickle", "rb") as input_file:
    df = pickle.load(input_file)
# print(df.columns)

periods = df["Period"].value_counts()
# cultures = df["Culture"].value_counts()
# dynasty = df["Dynasty"].value_counts()
department = df["Department"].value_counts()
# classification = df["Classification"].value_counts()
# object_names = df["Object Name"].value_counts()

edo = df.loc[df['Period'] == "Drawings and Prints"]
# edo = df.loc[df['Period'] == "Edo period (1615–1868)"]
# print(edo['Object Name'].value_counts()[:20])
print(periods[:20])
edo_prints = edo.loc[edo['Object Name'] == "Print"]
# edo_prints_by_ident = edo_prints["Is Public Domain"].value_counts()
# print(edo_prints["Culture"].value_counts())
# print(edo_prints_by_ident)
print(edo_prints["Object ID"])
# print(periods)

# object_names_edo = edo["Object Name"].value_counts()
# print(object_names_edo)

# print(edo_prints["Classification"].value_counts())
# print(edo_prints["Culture"].value_counts())

# print(department)

# for index,item in dynasty.iteritems():
#     if item > 5000:
#         print(index,item)


if __name__ == '__main__':
    # url = "https://collectionapi.metmuseum.org/public/collection/v1/objects/36461"
    # r = requests.get(url)
    # # # print(r)
    # # # print(r.json())
    # dic = r.json()
    # image_url = dic["primaryImage"]
    # # # print(dic[0])
    # wget.download(image_url,out='image_folder/test_name.jpg')


    # import time
    # list_to_loop = [0]*100

    # for i,item in enumerate(list_to_loop):
    #     if i%80 == 0:
    #         time.sleep(1)
    #     # Rest of code here
    pass