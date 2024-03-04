import timeit
import easynmt
from easynmt import EasyNMT

start = timeit.default_timer()

# opus-mt
# m2m_100_1.2b
# m2m_100_418M
model = EasyNMT('m2m_100_418M')

print(model.translate('学科：世界历史_古代中世纪_东北亚 定义：朝鲜现存的最古史书，1145年高丽王朝的学者金富轼(1075-1151)用古汉语撰成，记述了新罗、高句丽和百济三国的史事。 见载：《世界历史名词》第一版', target_lang='en'))
# expected translation: History of Three States in Ancient Korea

stop = timeit.default_timer()

print('Time: ', stop - start)