import requests
import json
import wget
if __name__ == '__main__':
    url = "https://collectionapi.metmuseum.org/public/collection/v1/objects/35972"
    r = requests.get(url)
    # print(r)
    # print(r.json())
    dic = r.json()
    image_url = dic["primaryImage"]
    # print(dic[0])
    wget.download(image_url,out='test_folder/test_name.jpg')


    import time
    list_to_loop = [0]*100

    for i,item in enumerate(list_to_loop):
        if i%80 == 0:
            time.sleep(1)
        # Rest of code here