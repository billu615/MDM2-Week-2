print('heleleo world hrh')
import json

words = json.load(open("dictionary.json"))
alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
entered = []

score = 1
high = 2
lives = 2
playing = True

print("\n\nWelcome to the Alphabet Game!\nEnter words, and see how long you can last without repeating!"
      "\nEnter 'qqq' to give up!")

i = 0

while playing:

    i += 1

    print("\nRound", str(i) + "!\n")

    for letter in alphabet:
        word = input("Enter a word starting with the letter " + letter.upper() + ": ")

        while word == "":
            word = input("Actually enter a word please: ")
        else:
            while playing:
                if word == "":
                    word = input("Please enter an actual word: ")
                else:
                    if word[0] != letter:
                        if word.lower() == "qqq":
                            if(input("\nAre you sure you want to quit? Enter 'qqq' again to quit: ")).lower() == "qqq":
                                print("\nUnfortunate!\nThanks for playing! Your highscore was:", high)
                                exit()
                            else:
                                word = input("\nPhew!\nEnter a word starting with the letter " + letter.upper() + ": ")
                                continue
                        word = input("Actually enter a word starting with the letter " + letter.upper() + ": ")
                    else:
                        if word not in words:
                            print("The word entered is not in the dictionary!")
                            word = input("Enter a word starting with the letter " + letter.upper() + ": ")
                        else:
                            if word in alphabet:
                                word = input("Enter an actual word please: ")
                            else:
                                break

        if word in entered:
            if lives == 0:
                playing = False
                if score > high:
                    high = score
                print("\nLost all lives!\nGame Over!\nWell played! Your highscore was:", high)
                break
            else:
                lives -= 1
                print("\nUh oh, you repeated a word!\nLife lost!\nLives left:", lives + 1, "\n")

        else:
            entered.append(word)
            score += 1
            if score > high:
                high = score

    if playing:
        print("\n\nCONGRATULATIONS!\n\nRound finished!\nCurrent score:", score, "\nLives left:", lives + 1)
        if score > high:
            high = score
    else:
        again = input("\nEnter Y to play again!\nAnswer: ")
        if again.upper() == "Y":
            playing = True
            entered = []
            score = 0
            i = 0
            lives = 2
        else:
            print("\nThanks for playing! Your highscore was:", high)
            quit()
