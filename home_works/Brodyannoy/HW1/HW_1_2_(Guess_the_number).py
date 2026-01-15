# HW1_2 "Guess the number"

import random

class GuessTheNumber:
    def __init__(self, low=1, high=10):
        self.low = low
        self.high = high
        self.previous_guess = None
        self.attempts = 0

    def hot_cold_feedback(self, guess):
        """
        Определяет 'горячо / холодно' относительно предыдущей попытки
        """
        if self.previous_guess is None:
            return "Начинаем игру"

        distance = abs(guess - self.previous_guess)

        if distance == 0:
            return "То же самое число"
        elif distance == 1:
            return "Очень горячо"
        elif distance <= 2:
            return "Горячо"
        else:
            return "Холодно"

    def make_guess(self):
        """
        Случайная попытка внутри текущего диапазона
        """
        return random.randint(self.low, self.high)

    def update_range(self, guess, answer):
        """
        Сужаем диапазон — бинарная логика
        """
        if answer == ">":
            self.low = guess + 1
        elif answer == "<":
            self.high = guess - 1

    def play(self):
        print("Загадайте число от 1 до 10")
        print("Отвечайте:")
        print("  >  если число больше")
        print("  <  если меньше")
        print("  =  если я угадал\n")

        while self.low <= self.high:
            guess = self.make_guess()
            self.attempts += 1

            print(f"Попытка №{self.attempts}: {guess}")
            print(self.hot_cold_feedback(guess))

            answer = input("Ваш ответ (> < =): ").strip()

            if answer == "=":
                print(f"\nЯ угадал число {guess} за {self.attempts} попыток!")
                return
            elif answer in (">", "<"):
                self.update_range(guess, answer)
                self.previous_guess = guess
            else:
                print("Введите только >, < или =")
                self.attempts -= 1

        print("\nПохоже, ответы были противоречивыми.")


if __name__ == "__main__":
    game = GuessTheNumber()
    game.play()

