import time
import json
from urllib.request import urlopen, quote
from geopy.distance import geodesic


def print_time():
    global time_1
    tem_t = time.time()
    tem = tem_t - time_1
    z = round(tem, 3)
    print("累计消耗时间为:" + str(z))


time_1 = time.time()


def getlnglat(address):
    url = 'http://api.map.baidu.com/geocoding/v3/'
    output = 'json'
    ak = 'UL71V0mIMUCRYWj24HQFqUgHoY67mdQ5'  # 应用时改为企业ak，其余都不需要修改
    add = quote(address)  # 由于本文城市变量为中文，为防止乱码，先用quote进行编码
    uri = url + '?' + 'address=' + add + '&output=' + output + '&ak=' + ak
    req = urlopen(uri)
    res = req.read().decode()  # 将其他编码的字符串解码成unicode
    temp = json.loads(res)  # 对json数据进行解析
    if temp['status'] != 0:
        return temp['status'], None
    lng = temp['result']['location']['lng']
    lat = temp['result']['location']['lat']  # 纬度——latitude,经度——longitude
    return lat, lng


def getlnglat_2(address):
    url = 'http://api.map.baidu.com/geocoding/v3/'
    output = 'json'
    ak = 'UL71V0mIMUCRYWj24HQFqUgHoY67mdQ5'
    add = quote(address)  # 由于本文城市变量为中文，为防止乱码，先用quote进行编码
    uri = url + '?' + 'address=' + add + '&output=' + output + '&ak=' + ak
    req = urlopen(uri)
    res = req.read().decode()  # 将其他编码的字符串解码成unicode
    temp = json.loads(res)  # 对json数据进行解析

    return temp


address_1 = '北京大学国际医院'
address_2 = '北京大学第三医院（北医三院）'

tuple_1 = getlnglat(address_1)
tuple_2 = getlnglat(address_2)

print_time()
print(geodesic(tuple_1, tuple_2).km) # 参数为两个tuple，每个tuple都是先lat后lng，返回单位为km

# print(getlnglat_2(address_1))
# print(getlnglat_2(address_2))

print_time()
# print(getlnglat_2(address_1))
