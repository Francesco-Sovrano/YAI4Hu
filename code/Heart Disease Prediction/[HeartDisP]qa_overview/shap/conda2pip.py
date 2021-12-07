import ruamel.yaml
import sys
import os

conda_yaml = sys.argv[1]
destination_dir = sys.argv[2]
yaml = ruamel.yaml.YAML()
data = yaml.load(open(conda_yaml))

requirements = []
for dep in data['dependencies']:
    if isinstance(dep, str):
        package, package_version, python_version = dep.split('=')
        if python_version == '0':
            continue
        requirements.append(package + '==' + package_version)
    elif isinstance(dep, dict):
        for preq in dep.get('pip', []):
            requirements.append(preq)

with open(os.path.join(destination_dir,'requirements.txt'), 'w') as fp:
    for requirement in requirements:
       print(requirement, file=fp)