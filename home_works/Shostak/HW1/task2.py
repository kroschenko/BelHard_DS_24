#игра "Угадай число"
while True:
    print("Загадай число от 1 до 10")
    min = 1
    max = 10

    while True:
        number=(min+max) // 2
        print("Твое число:",number)
        answer=input("Введи '>' если твое число больше, '<' если твое число меньше,'=' если угадал:")
        if answer == "=":
            print("Твое число:", number)
            break
        elif answer == ">":
            min = number + 1
        elif answer == "<":
            max = number - 1

    play_again = input("Хочешь сыграть еще раз?(да/нет):")
    if play_again != "да":
        print("Пока!")
        break

