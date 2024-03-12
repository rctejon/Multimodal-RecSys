import json
from tqdm import tqdm
import math
import pickle
import os
from datetime import datetime


def calculate_rating(progress, threshold=0.8):
    if progress >= threshold:
        return 5
    else:
        return 5 * math.sqrt(progress / threshold)


interactions_file_name = "data/MOOCCubeX/relations/user-video.json"
course_file_name = "data/MOOCCubeX/entities/course.json"
problems_file_name = "data/MOOCCubeX/relations/user-problem.json"
exercises_file_name = "data/MOOCCubeX/relations/exercise-problem.txt"

users = open(interactions_file_name, encoding='utf-8')
courses = open(course_file_name, encoding='utf-8')
problems = open(problems_file_name, encoding='utf-8')
exercises = open(exercises_file_name, encoding='utf-8')

count = 0
resource_course = {}
course_resources = {}

for course in courses:
    course = json.loads(course)

    course_resources[course['id']] = []

    for resource in course['resource']:
        resource_course[resource['resource_id']] = course['id']
        course_resources[course['id']].append(resource['resource_id'])

courses.close()

problem_exercise = {}
for exercise in tqdm(exercises, total=6251341):
    exercise = exercise.split()
    problem_exercise[exercise[1]] = exercise[0]

user_exercise = {}

if not os.path.exists('pickles/user_exercise.pkl'):
    for problem in tqdm(problems, total=133384333):
        problem = json.loads(problem)
        # count += 1
        if problem['problem_id'] in problem_exercise.keys():
            if problem['user_id'] not in user_exercise.keys():
                user_exercise[problem['user_id']] = {'exercises': set(), 'time': int(datetime.strptime(problem['submit_time'], "%Y-%m-%d %H:%M:%S").timestamp())}
            user_exercise[problem['user_id']]['exercises'].add(problem_exercise[problem['problem_id']])
        # if count == 1000000:
        #     break

    with open('pickles/user_exercise.pkl', 'wb') as f:
        pickle.dump(user_exercise, f)
else:
    with open('pickles/user_exercise.pkl', 'rb') as f:
        user_exercise = pickle.load(f)
problems.close()

user_ratings8 = {}
user_ratings5 = {}
user_ratings3 = {}
user_ratings2 = {}

for user in tqdm(users, total=310360):
    user = json.loads(user)
    user_video = [(video['video_id'], video['segment'][0]['local_start_time'])for video in user['seq']]
    user_courses = {}
    for video_time in user_video:
        video = video_time[0]
        if video in resource_course.keys():
            course = resource_course[video]
            if course not in user_courses.keys():
                user_courses[course] = {'resources': [video], 'time': video_time[1]}
            else:
                user_courses[course]['resources'].append(video)
                user_courses[course]['time'] = min(video_time[1], user_courses[course]['time'])
        

    if user['user_id'] in user_exercise.keys():
        for exercise in user_exercise[user['user_id']]['excercises']:
            if exercise in resource_course.keys():
                course = resource_course[exercise]
                if course not in user_courses.keys():
                    user_courses[course] = {'resources': [exercise], 'time': user_exercise[user['user_id']]['time']}
                else:
                    user_courses[course]['resources'].append(exercise)
                    user_courses[course]['time'] = min(user_courses[course]['time'], user_exercise[user['user_id']]['time'])

    if len(user_courses):
        user_course_progress = {course: {'progress': len(videos['resources']) / len(course_resources[course]), 'time': videos['time']} for course, videos in
                                user_courses.items()}
        user_ratings5[user['user_id']] = [{'course': course,
                                                    'rating': calculate_rating(progress['progress'], threshold=0.5), 
                                                    'time': progress['time']
                                                    } for course, progress in
                                          user_course_progress.items()
                                         ]

users.close()

with open(f'pickles/user_ratings5.pkl', 'wb') as f:
    pickle.dump(user_ratings5, f)

print('0.5:', len(user_ratings5))



