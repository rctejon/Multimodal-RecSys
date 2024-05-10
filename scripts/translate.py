import timeit
from easynmt import EasyNMT
import json
from tqdm import tqdm
import pickle


def join_fields(fields):
    translate_text = ''
    for i in range(len(fields)):
        translate_text += f'Topic {i + 1}: {fields[i]}\n'
    return translate_text


def translate_fields(fields):
    translated_fields = []
    for field in fields:
        try:
            translated_fields.append(model.translate(field, target_lang='en'))
        except:
            continue
        # print(translated_fields[-1])
    return translated_fields


# opus-mt
# m2m_100_1.2b
# m2m_100_418M
model = EasyNMT('m2m_100_418M')

# print(model.translate('学科：世界历史_古代中世纪_东北亚 定义：朝鲜现存的最古史书，1145年高丽王朝的学者金富轼(1075-1151)用古汉语撰成，记述了新罗、高句丽和百济三国的史事。 见载：《世界历史名词》第一版', target_lang='en'))
# expected translation: History of Three States in Ancient Korea

start = timeit.default_timer()

course_file_name = "../data/MOOCCubeX/entities/course.json"

courses = open(course_file_name, encoding='utf-8')

course_texts = {}

# count = 0

for course in tqdm(courses, total=3781):

    # if count < 28:
    #     count += 1
    #     continue

    course = json.loads(course)

    translated_fields = translate_fields(course['field'])

    try:
        translated_about = model.translate(course['about'], target_lang='en') if course['about'] is not None else ''
    except:
        translated_about = ''

    text = join_fields(translated_fields) + f'Description: {translated_about}'
    course_texts[course['id']] = text

courses.close()

stop = timeit.default_timer()

print('Time: ', stop - start)

pickle.dump(course_texts, open('../data/MOOCCubeX/course_texts.pkl', 'wb'))

pickle.load(open('../data/MOOCCubeX/course_texts.pkl', 'rb'))
