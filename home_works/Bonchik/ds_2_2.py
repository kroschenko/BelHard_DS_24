class BinaryGuessGame:
    def __init__(self, min_num=1, max_num=10):
        self.min_num = min_num
        self.max_num = max_num

    def play(self):
        print(f"Загадайте число от {self.min_num} до {self.max_num}. Я буду угадывать!")
        low, high = self.min_num, self.max_num
        steps = 0

        while low <= high:
            # Проверка: если остался один вариант
            if low == high:
                print(f"Загаданное число: {low}")
                print(f"Я угадал за {steps} шаг(а/ов) 🎉")
                break

            steps += 1
            guess = (low + high) // 2
            print(f"Мой вариант: {guess}")

            answer = input("Это число? (1 - да, 2 - нет): ").strip().lower()
            if answer in ("1", "да"):
                print(f"Ура! Я угадал за {steps} шаг(а/ов) 🎉")
                break
            elif answer in ("2", "нет"):
                hint = input("Загаданное число меньше или больше? (1 - меньше, 2 - больше): ").strip().lower()
                if hint in ("1", "меньше"):
                    high = guess - 1
                elif hint in ("2", "больше"):
                    low = guess + 1
                else:
                    print("Введите только '1/меньше' или '2/больше'.")
            else:
                print("Введите только '1/да' или '2/нет'.")

        again = input("Хотите сыграть ещё раз? (1 - да, 2 - нет): ").strip().lower()
        if again in ("1", "да"):
            self.play()


if __name__ == "__main__":
    game = BinaryGuessGame()
    game.play()
