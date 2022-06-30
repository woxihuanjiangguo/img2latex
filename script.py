with open('./data-beam/train.log', 'r') as file:
    lines = file.readlines()
    for line in lines:
        if 'val' in line:
            splits = line.split(' ')
            print(splits[1].strip('\n'))