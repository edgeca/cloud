"""Send prediction requests to OD API.
"""

import argparse
import random
from multiprocessing import Pool
from statistics import mean
from timeit import default_timer as timer

import requests

url = 'http://127.0.0.1:9696'
models_pool = ['infosys_tables']
data_pool = ['/jmeter/36_images/']


def predict(model_name, images):
    try:
        start = timer()
        print("Sending request for model: {}, with images(s): {}".format(
            model_name, images))
        r = requests.post(url + '/predict', verify=False, data={
            'input_path': images,
            'model': model_name
        })
        response = r.json()
        if response['status'] is None or response['status'] != 'success':
            raise Exception
        end = timer()
        print("Completed prediction for model: {}, with image(s): {} in {:0.2f}s".format(
            model_name, images, end - start))
        return end - start
    except Exception as e:
        print(e)


def get_model_names(concurrent_models=1, total_models=1):
    model_names = []
    sampled_models = random.sample(models_pool, concurrent_models)
    if total_models > concurrent_models:
        residual_models = total_models - concurrent_models
        for i, model_name in enumerate(sampled_models):
            if i == concurrent_models - 1:
                model_count = 1 + residual_models
            else:
                model_count = 1 + random.randint(0, residual_models)
                residual_models = residual_models - model_count + 1
            model_names.extend([model_name] * model_count)
    else:
        model_names = sampled_models
    return model_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--url', help='URL of OD API', default='https://vision-api-service:443', type=str)
    parser.add_argument(
        '--users', help='Number of concurrent users', default=1, type=int)
    parser.add_argument(
        '--models', help='Number of concurrent models', default=1, type=int)
    parser.add_argument('--size', help='Batch size', default=0, type=int)
    parser.add_argument('--iterations', help='Iterations', default=1, type=int)
    args = parser.parse_args()
    users = args.users
    num_models = args.models
    url = args.url

    start = timer()
    for i in range(args.iterations):
        model_names = get_model_names(
            concurrent_models=num_models, total_models=users)
        inputs = [random.choice(
            data_pool) if args.size <= 0 else str(args.size) for user in range(users)]
        with Pool(users) as p:
            runtimes = p.starmap(predict, zip(
                model_names, inputs))
        print("Min. response time: {:0.2f}s".format(min(runtimes)))
        print("Max. response time: {:0.2f}s".format(max(runtimes)))
        print("Avg. response time: {:0.2f}s".format(mean(runtimes)))
        print("Completed iteration: {}".format(i+1))

    end = timer()
    print("Completed script execution in {:0.2f}s".format(end - start))
