import random
from collections import deque
import time
from enum import Enum

# Codes for each symbol
heart = "\u2665"
spade = "\u2660"
diamond = "\u2666"
club = "\u2663"

# Set the suits to each symbol
suits = {
    "diamonds": diamond,
    "hearts": heart,
    "spades": spade,
    "clubs": club
}


# Set up the Blackjack class
class BlackJackGame:
    # Set up game status
    class GameStatus(Enum):
        CONTINUE = 1
        WIN = 2
        BUST = 3
        DRAW = 4
        BLACKJACK = 5
        LOSE = 6

    # Create a deck to pull cards from
    @staticmethod
    def generate_deck():
        numbers = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        deck = [{'number': number, 'suit': suit} for number in numbers for suit in suits]
        return deck

    # Format the cards in each hand using the symbol to show the suit
    @staticmethod
    def format_cards(cards):
        result = ""
        for card in cards:
            suit = suits[card["suit"]]
            result += f"{card['number']}{suit} "
        return result.strip()

    def __init__(self):
        self.deck = deque(self.generate_deck())  # generate a deck of cards
        random.shuffle(self.deck)  # Shuffle the deck
        self.player_hand = {0: []}  # Tracks which hand the player is currently on, with the cards inside each hand
        self.player_hand_value = {0: 0}  # Tracks each of the player's hand values
        self.player_aces = {0: 0}  # Tracks how many aces the player has for each hand
        self.usable_ace = False
        self.current_hand_index = 0  #
        self.dealer_hand = []  # Tracks the cards inside the dealers hand
        self.dealer_value = 0  # Tracks the dealers value
        self.dealer_aces = 0  # Track how many aces the dealer has
        self.status = {0: self.GameStatus.CONTINUE}  # Used to track the games status across all hands
        # Precompute card values to avoid recalculating them every time `update_hand_value` is called.
        self.card_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                            '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 11}

    # Use self.status to determine when the game ends etc
    def game_status(self, hand_index):
        # Check hand value for bust
        if self.player_hand_value[hand_index] > 21:
            self.status[hand_index] = self.GameStatus.BUST
            return self.status[hand_index]
        # Else continue
        else:
            self.status[hand_index] = self.GameStatus.CONTINUE
            return self.status[hand_index]

    # Update each hand value, use is_player to determine which hand to update
    def update_hand_value(self, card, hand_index):
        value = self.card_values[card['number']]
        # Check if player hand or dealer hand
        if hand_index is not None:
            # Update player's hand value
            self.player_hand_value[hand_index] += value
            if card['number'] == 'A':
                self.player_aces[hand_index] += 1
                self.usable_ace = True
            while self.player_hand_value[hand_index] > 21 and self.player_aces[hand_index] > 0:
                self.player_hand_value[hand_index] -= 10
                self.player_aces[hand_index] -= 1
                self.usable_ace = False
        else:
            # Update dealer's hand value
            self.dealer_value += value
            if card['number'] == 'A':
                self.dealer_aces += 1
            while self.dealer_value > 21 and self.dealer_aces > 0:
                self.dealer_value -= 10
                self.dealer_aces -= 1

    # Deal a card to the players hand and update the hand value using the above function
    def player_deal(self, hand_index):
        # Remove a card from the deck and set the new card to that card's value
        card = self.deck.pop()
        # Add the newly drawn card to the player's current hand
        self.player_hand[hand_index].append(card)
        # Update the player's current hand value with the new card's value
        self.update_hand_value(card, hand_index)
        return card

    # Deal a card to the dealer and update the dealer's hand value
    def dealer_deal(self):
        # Remove a card from the deck and set the new card to that card's value
        card = self.deck.pop()
        # Add the newly drawn card to the dealers hand
        self.dealer_hand.append(card)
        # Update the dealer's hand value with the new card's value
        self.update_hand_value(card, None)
        return card

    # Gets the players action for each hand in the hand index
    @staticmethod
    def get_player_action(hand_index):
        while True:
            action = input(f"Enter an action for hand {hand_index + 1} (hit, stay, or split): ").lower()
            if action in ["hit", "stay", "split"]:
                return action
            print("Please choose to either hit, stay, split or double down")

    # This function allows the player to choose what action they want to take(hit / stay)
    def player_action(self, action, hand_index):
        if action == "hit":
            self.player_deal(hand_index)  # Deal a card to the player's current hand
            print(f'Hand {hand_index + 1}: {self.format_cards(self.player_hand[hand_index])}, Total Hand Value: '
                  f'{self.player_hand_value[hand_index]}')
            return self.game_status(hand_index)  # Update status after adding a card to hand to check if player busts
        elif action == "stay":
            print("Player Stays")
            return self.game_status(hand_index)  # Continue the game after the player decides they stay
        elif action == "split":
            # Confirm eligible split (2 cards in hand of the same 'number')
            if (len(self.player_hand[hand_index]) == 2 and self.player_hand[hand_index][0]['number'] ==
                    self.player_hand[hand_index][1]['number']):
                self.split_hand(hand_index)
            else:
                print("You cannot split this hand")

    # This function handles the splitting logic
    def split_hand(self, hand_index):
        # Create a new hand index for the split hand and keep track of the card were splitting with
        new_hand_index = max(self.player_hand.keys()) + 1
        card = self.player_hand[hand_index].pop()

        # Subtract the card's value from the original hand's value
        card_value = self.card_values[card['number']]
        self.player_hand_value[hand_index] -= card_value
        if card['number'] == 'A':
            self.player_aces[hand_index] -= 1

        # Initialize the new hand
        self.player_hand[new_hand_index] = [card]
        self.player_hand_value[new_hand_index] = card_value
        self.player_aces[new_hand_index] = 1 if card['number'] == 'A' else 0
        self.status[new_hand_index] = self.GameStatus.CONTINUE

        # Deal a new card to each split hand and update/print the values
        # First hand
        self.player_deal(hand_index)
        print(f'Hand {hand_index + 1}: {self.format_cards(self.player_hand[hand_index])}, Total Hand Value: '
              f'{self.player_hand_value[hand_index]}')
        # Second hand
        self.player_deal(new_hand_index)
        print(f'Hand {new_hand_index + 1}: {self.format_cards(self.player_hand[new_hand_index])}, Total Hand Value: '
              f'{self.player_hand_value[new_hand_index]}')

    # Handles dealer's turn
    def dealer_action(self, output=True):
        while self.dealer_value < 17:
            self.dealer_deal()
            if output:
                print("Dealer hits and has: ", self.format_cards(self.dealer_hand), self.dealer_value)
                #time.sleep(1)  # Change this later when using the bot

    # Handles the start of the game mechanics
    def start_game(self):
        self.player_deal(0)
        self.player_deal(0)
        self.dealer_deal()
        self.dealer_deal()
        print("Dealer shows:", self.format_cards(self.dealer_hand[:1]))  # Show only the first card of the dealer's hand
        print("Player's Cards: ", self.format_cards(self.player_hand[0]), self.player_hand_value[0])
        if self.player_hand_value[0] == 21:
            self.status[0] = self.GameStatus.BLACKJACK
            print(f"Player BlackJack!")
            return self.status[0]

    # This function allows the game to continue until the player quits, resetting each of the variables
    def reset(self):
        self.deck = deque(self.generate_deck())  # generate a deck of cards
        random.shuffle(self.deck)  # Shuffle the deck
        self.player_hand = {0: []}  # Tracks which hand the player is currently on, with the cards inside each hand
        self.player_hand_value = {0: 0}  # Tracks each of the player's hand values
        self.player_aces = {0: 0}  # Tracks how many aces the player has for each hand
        self.current_hand_index = 0  #
        self.dealer_hand = []  # Tracks the cards inside the dealers hand
        self.dealer_value = 0  # Tracks the dealers value
        self.dealer_aces = 0  # Track how many aces the dealer has
        self.status[0] = self.GameStatus.CONTINUE

    # Entire game logic
    def play_round(self):
        # Initialize player and dealer hand
        self.start_game()
        hand_indices = list(self.player_hand.keys())
        i = 0
        while i < len(hand_indices):
            hand_index = hand_indices[i]
            self.current_hand_index = hand_index
            if self.status[hand_index] == self.GameStatus.BLACKJACK:
                i += 1
                continue
            else:
                self.status[hand_index] = self.GameStatus.CONTINUE  # Reset the status for each hand
            while self.status[hand_index] == self.GameStatus.CONTINUE:
                action = self.get_player_action(hand_index)
                if action == "stay":
                    break
                self.player_action(action, hand_index)
                if action == "split":
                    # Update hand_indices to include the new hand created by splitting
                    hand_indices = list(self.player_hand.keys())
            if self.status[hand_index] == self.GameStatus.BUST:
                print(
                    f"Hand {hand_index + 1}: Player Bust and lost")
            i += 1

        # Dealer's turn
        print(f"Dealer reveals second card: {self.format_cards(self.dealer_hand)} {self.dealer_value}")
        self.dealer_action(output=True)

        result = self.game_result()
        print(result)

    # Logic to get the result of the game for each hand
    def game_result(self):
        dealer_value = self.dealer_value  # Set a local variable

        # Iterate over each hand to determine the results of the game
        for hand_index, player_value in self.player_hand_value.items():
            # Check if the player busts
            if self.player_hand_value[hand_index] > 21:
                result = "Player Bust"
                return result
            # Check if the player has blackjack
            elif self.status[hand_index] == self.GameStatus.BLACKJACK:
                result = "Blackjack"
                return result
            elif dealer_value > 21:
                result = "Dealer Bust"
                return result
            elif player_value > dealer_value:
                result = "Player Value Greater"
                return result
            elif player_value == dealer_value:
                result = "Draw"
                return result
            else:
                result = "Dealer Value Greater"
                return result


game = BlackJackGame()


def main():
    game.play_round()
# Turn into pygame GUI later, or just use this to teach the AI and use it to give the AI cards and get the approx value


if __name__ == "__main__":
    main()
