


# write a function that receives a string and prints  output either red, yellow or green
def cprint(string, color):
    if color == 'red':
        print(f'\033[31m{string}\033[0m')
    elif color == 'yellow':
        print(f'\033[33m{string}\033[0m')
    elif color == 'green':
        print(f'\033[32m{string}\033[0m')
    else:
        print(string)