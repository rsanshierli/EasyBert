import hashlib
import urllib
import json
from translate import Translator
import random



class convertText(object):
    def __init__(self, fromLangByBaidu, toLangByBaidu, fromLangByMicrosoft, toLangByMicrosoft):
        self.appid = '' # 填写你的appid
        self.secretKey = '' # 填写你的密钥
        self.url_baidu_api = 'http://api.fanyi.baidu.com/api/trans/vip/translate'  # 百度通用api接口
        self.fromLang = fromLangByBaidu
        self.toLang = toLangByBaidu
        self.fromLangByMicrosoft = fromLangByMicrosoft
        self.toLangByMicrosoft = toLangByMicrosoft
        # self.stop_words = self.load_stop_word(os.path.join(sys.path[0], 'stop_words.txt'))

    def _translateFromBaidu(self, text, fromLang, toLang):
        salt = random.randint(32768, 65536)  # 随机数
        sign = self.appid + text + str(salt) + self.secretKey  # 签名 appid+text+salt+密钥
        sign = hashlib.md5(sign.encode()).hexdigest()  # sign 的MD5值

        url_baidu = self.url_baidu_api + '?appid=' + self.appid + '&q=' + urllib.parse.quote(
            text) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(salt) + '&sign=' + sign
        #        进程挂起时间1s
        #        time.sleep(1)

        try:
            response = urllib.request.urlopen(url_baidu, timeout=30)
            content = response.read().decode("utf-8")
            data = json.loads(content)

            if 'error_code' in data:
                print('错误代码：{0}, {1}'.format(data['error_code'], data['error_msg']))
                return 'error'
            else:
                return str(data['trans_result'][0]['dst'])
        except urllib.error.URLError as error:
            print(error)
            return 'error'
        except urllib.error.HTTPError as error:
            print(error)
            return 'error'


    # 使用百度翻译API进行回译 chinese->english->chinese
    def convertFromBaidu(self, text):
        translation1 = self._translateFromBaidu(text, self.fromLang, self.toLang)
        if translation1 == 'error':
            return 'error'
        translation2 = self._translateFromBaidu(translation1, self.toLang, self.fromLang)
        if translation2 == 'error':
            return 'error'
        if translation2 != text:
            return translation2

        return 'same'

    # 使用微软翻译API进行回译 chinese->english->chinese
    def convertFromMicrosoft(self, text):
        translator1 = Translator(from_lang=self.fromLangByMicrosoft, to_lang=self.toLangByMicrosoft)
        translation1 = translator1.translate(text)

        translator2 = Translator(from_lang=self.toLangByMicrosoft, to_lang=self.fromLangByMicrosoft)
        translation2 = translator2.translate(translation1)

        if translation2 != text:
            return translation2

        return 'same'

if __name__ == '__main__':
    text = '电影评价,窗前明月光，我很喜十八你欢这部电影！你和啊哈哈呢，我的宝贝？'
    conText = convertText(fromLangByBaidu='zh',toLangByBaidu='en',fromLangByMicrosoft='chinese',toLangByMicrosoft='english')
    #
    print(conText.convertFromBaidu(text))
    # print(conText.convertFromMicrosoft())