# Game Playing Agent
Adversarial Game Playing Agent

# Initial game setup for two players
- Simple wooden pawn-style playing pieces, often called "Halma pawns."
- The board consists of a grid of 16×16 squares.
- Each player's camp consists of a cluster of adjacent squares in one corner of the board.
These camps are delineated on the board.
- For two-player games, each player's camp is a cluster of 19 squares. The camps are in
opposite corners.
- Each player has a set of pieces in a distinct color, of the same number as squares in each
camp.
- The game starts with each player's camp filled by pieces of their own color.

# Game rules
Create the initial board setup according to the above description.
- Players randomly determine who will move first.
- Pieces can move in eight possible directions (orthogonally and diagonally).
- Each player's turn consists of moving a single piece of one's own color in one of the
following plays:<br>
  - One move to an empty square:<br>
    - Move the piece to an empty square that is adjacent to the piece’s original
position (with 8-adjacency).
    - This move ends the play for this player’s turn.
  - One or more jumps over adjacent pieces:
    - An adjacent piece of any color can be jumped if there is an empty square
on the directly opposite side of that piece.
    - Place the piece in the empty square on the opposite side of the jumped
piece.
    - The piece that was jumped over is unaffected and remains on the board.
    - After any jump, one may make further jumps using the same piece, or end
the play for this turn.
    - In a sequence of jumps, a piece may jump several times over the same
other piece.
- Once a piece has reached the opposing camp, a play cannot result in that piece leaving
the camp.
- If the current play results in having every square of the opposing camp that is not already
occupied by the opponent to be occupied by one's own pieces, the acting player wins.
Otherwise, play proceeds to the other player.
