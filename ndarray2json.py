import numpy as np
import json

choice = int(input("輸入0進行 ndarray2json, 輸入1進行 json2ndarray : "))

if choice == 0:

    array = np.array(eval(input("輸入ndarray:")))

    array_json = json.dumps(array.tolist(), ensure_ascii=False, indent=4)

    with open('array.json', 'w', encoding= 'utf-8') as json_file:
        json_file.write(array_json)

else:

    file_path = str(input("輸入 json 檔路徑:"))

    with open(file_path, 'r', encoding= 'utf-8') as json_file:
        array_json = json_file.read()

    array = np.array(json.loads(array_json))

    print(repr(array))



#[{'message': '整體餐不錯吃，肉料理的很剛好，食材新鮮！', 'starNumber': '5'},{'message': '很喜歡餐點跟氣氛，很棒，下次還會再去', 'starNumber': '5'},{'message': '中午11點 到12點人潮較多', 'starNumber': '4'},{'message': '我覺得很好， 真的很好。', 'starNumber': '5'},{'message': '獲贈餅乾一枚🥰', 'starNumber': '5'},{'message': '環境好，洽公商談的好去處。', 'starNumber': '5'},{'message': '悠閒的用餐休息區。', 'starNumber': '4'},{'message': '整個套餐吃下來中規中矩，好吃但不驚艷。', 'starNumber': '4'},{'message': '食物美味，可以包場，很讚的聚餐場所！', 'starNumber': '4'},{'message': '滿特別的料理，賞心悅目', 'starNumber': '4'},{'message': '值得每季都來一回的好餐廳😋😋', 'starNumber': '5'},{'message': '菜色棒 服務親切酒水價格也很合理', 'starNumber': '5'},{'message': '人員的服務很好👍🏻餐點也很好吃', 'starNumber': '5'},{'message': '東西好吃 味道很有特色', 'starNumber': '5'},{'message': '好吃， 服務好。', 'starNumber': '5'},{'message': '令人驚艷，服務親切疫情期間經過這裡 …', 'starNumber': '5'},{'message': '環境很好、食材新鮮、價格合宜,推薦 ！', 'starNumber': '5'},{'message': '食材新鮮，服務非常好❤️', 'starNumber': '5'},{'message': '食材新鮮好吃，服務周到大大的滿足！！', 'starNumber': '5'},{'message': '食材新鮮 服務很好用餐環境很棒🎉', 'starNumber': '5'}]
