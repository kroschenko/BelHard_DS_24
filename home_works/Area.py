class Rectangle:
    def __init__(self,side_a: int,side_b: int):
        self.side_a = side_a
        self.side_b = side_b

    def area(self) -> int:
        return self.side_a * self.side_b


a: int = int(input("Enter the first side: "))
b: int = int(input("Enter the second side: "))
rect = Rectangle(a,b)

print("Area of Rectangle: ", rect.area())