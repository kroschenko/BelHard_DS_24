import random

class GuessTheNumberGame:
    def __init__(self):
        self.a = 0
        self.b = 0
        self.user_number = 0
        self.pc_number = 0
        self.count = 0

    def ask_to_play(self) -> bool:
        while True:
            choice = input("Do you wanna play in guess the number? (1-yes, 2-no): ").lower()
            if choice in ("1", "yes"):
                return True
            if choice in ("2", "no"):
                return False
            print("Invalid input. Try again.")

    def setup_range(self):
        self.a = int(input("Select from: "))
        self.b = int(input("Select to: "))

        if self.a > self.b:
            self.a, self.b = self.b, self.a

    def get_user_number(self):
        while True:
            num = int(input(f"Enter guess number from {self.a} to {self.b}: "))
            if self.a <= num <= self.b:
                self.user_number = num
                return
            print("Invalid input. Guess the number.")

    def pc_guess(self):
        self.pc_number = random.randint(self.a, self.b)

    def play_round(self):
        self.count = 0
        self.pc_guess()

        while True:
            self.count += 1

            if self.pc_number == self.user_number:
                print("\nPC guessed the number!")
                print("Number of count:", self.count)
                print("Your number:", self.pc_number)
                return

            print(f"\nPC guessed {self.pc_number} and missed.")
            answer = input("Is your number greater? (1-yes, 2-no): ").lower()

            if answer in ("1", "yes"):
                self.a = self.pc_number + 1
            elif answer in ("2", "no"):
                self.b = self.pc_number - 1
            else:
                print("Invalid input. Try again.")
                self.count -= 1
                continue

            if self.a > self.b:
                print("You messed up somewhere. Range is impossible now.")
                return

            self.pc_number = (self.a + self.b) // 2

    def play(self):
        while self.ask_to_play():
            self.setup_range()
            self.get_user_number()
            print("\nPC is guessing the number...")
            self.play_round()

        print("\nBye!")



game = GuessTheNumberGame()
game.play()