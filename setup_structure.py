import os 

def create_project_structure(base_path):
    #create the proect stucture
    folders = [
        'data/raw',
        'data/transformed',
        'images',
        'models',
        'notebooks'
    ]

    #create sub directories
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    print(f'Project structure created succesfully at {base_path}')

#define the base path where the porject will be created
base_path = os.path.dirname(os.path.abspath(__file__))

#create the prohect structure
create_project_structure(base_path)

#in the terminal, type python setup_structure.py and create the project structure