#!C:\Users\User\AppData\Local\Programs\Python\Python38-32
import cgitb
import turtle
import time
import random
cgitb.enable()
start_response('200 OK', [('Content-Type', 'text/html')])
score = 0
high_score = 0
delay = 0.1

#setting up the scree
wndw = turtle.Screen()
wndw.title("The Snake Game")
wndw.bgcolor("black")
wndw.setup(width=700, height=700)
wndw.tracer(0)
segment = []

#writting the gameover
over = turtle.Turtle()
over.color("white")
over.speed(0)
over.penup()
over.shape("triangle")
over.hideturtle()
over.goto(0, 10)
#over.write("thanks for playing click arrows to restart", align = "center", font=("Arial", 20, 'bold'))

#writting the score
pen = turtle.Turtle()
pen.speed(0)
pen.color("white")
pen.shape("triangle")
pen.penup()
pen.hideturtle()
pen.goto(0,310)
pen.write("Score: 0   High Score: 0", align = "center", font=("Arial", 25, 'italic'))

#setting up the snake
head = turtle.Turtle()
head.speed(0)
head.color("white")
head.shape("circle")
head.penup()
head.goto(0, 0)
head.direction = "stop"

#setting snakes food
food = turtle.Turtle()
food.speed(0)
food.color("red")
food.shape("square")
food.penup()
food.goto(0, 100)

#functions used
def move_up():
    if head.direction != "down":
        over.clear()
        head.direction = "up"
def move_down():
    if head.direction != "up":
        over.clear()
        head.direction = "down"
def move_left():
    if head.direction != "right":
        over.clear()
        head.direction = "left"
def move_right():
    if head.direction != "left":
        over.clear()
        head.direction = "right"

#keyboard key bindings with game
wndw.listen()
wndw.onkeypress(move_up, "Up")
wndw.onkeypress(move_down, "Down")
wndw.onkeypress(move_left, "Left")
wndw.onkeypress(move_right, "Right")



def move():
    if head.direction == "up":
        y = head.ycor()
        head.sety(y + 20)
    if head.direction == "down":
        y = head.ycor()
        head.sety(y - 20)
    if head.direction == "left":
        x = head.xcor()
        head.setx(x - 20)
    if head.direction == "right":
        x = head.xcor()
        head.setx(x + 20)
#game loop
while True:
    wndw.update()

    #if snake touches the corner
    if head.xcor() > 340 or head.xcor() < (-340) or head.ycor() > 340 or head.ycor() < (-340):
        time.sleep(1)
        head.goto(0, 0)
        head.direction = "stop"
        delay = 0.1
        over.write("thanks for playing click arrows to restart", align = "center", font=("Arial", 20, 'bold'))

        #clearing the scores when it touches
        pen.clear()
        score = 0
        pen.write("Score: 0   High Score: {}".format(high_score), align = "center", font=("Arial", 25, 'italic'))

        #as snake touches the border we need to make the snake as it was in beginning (i.e. only head)
        for s in segment:
            s.goto(1000,1000)
        segment.clear()
    #when snake touches the food
    if head.distance(food) < 20:
        y = random.randint(-340, 340)
        x = random.randint(-340, 340)
        food.goto(x, y)

        #appending body to snake as it eats food
        new_segment = turtle.Turtle()
        new_segment.speed(0)
        new_segment.shape("circle")
        new_segment.color("grey")
        new_segment.penup()
        segment.append(new_segment)

        #updating score as snake eats food
        score+=10
        if score > high_score:
            high_score+=10
        pen.clear()
        pen.write("Score: {}   High Score: {}".format(score, high_score), align = "center", font=("Arial", 25, 'italic'))
        delay = 0.1

    #move the body of the snake behind its head
    for i in range(len(segment)-1,0,-1):
        x = segment[i-1].xcor()
        y = segment[i-1].ycor()
        segment[i].goto(x,y)

    #for the part after the head as it will not get concidered in for loop
    if len(segment) > 0:
        x = head.xcor()
        y = head.ycor()
        segment[0].goto(x, y)

    move()

    #if snake touches its body
    for s in segment:
        if s.distance(head) < 20:
            time.sleep(1)
            head.goto(0, 0)
            head.direction = "stop"
            for s in segment:
                s.goto(1000,1000)
            segment.clear()
            over.write("thanks for playing click arrows to restart", align = "center", font=("Arial", 20, 'bold'))
            pen.clear()
            score = 0
            pen.write("Score: 0   High Score: {}".format(high_score), align = "center", font=("Arial", 25, 'italic'))
            delay = 0.1

    time.sleep(delay)

wndw.mainloop()
