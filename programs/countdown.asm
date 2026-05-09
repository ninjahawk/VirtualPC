; countdown.asm — read N, print N, N-1, ..., 1, 0, then "blastoff!".
; $80 = current value
.org $00
    LDA #$6E      ; 'n'
    OUT
    LDA #$3A
    OUT
    LDA #$20
    OUT
    INP
    STA $80

loop:
    LDA $80
    OUTN
    LDA #$0A
    OUT

    LDA $80
    CMP #0
    JZ  blast

    LDA $80
    SUB #1
    STA $80
    JMP loop

blast:
    LDA #$62      ; 'b'
    OUT
    LDA #$6C      ; 'l'
    OUT
    LDA #$61      ; 'a'
    OUT
    LDA #$73      ; 's'
    OUT
    LDA #$74      ; 't'
    OUT
    LDA #$6F      ; 'o'
    OUT
    LDA #$66      ; 'f'
    OUT
    LDA #$66      ; 'f'
    OUT
    LDA #$21      ; '!'
    OUT
    LDA #$0A
    OUT
    HLT
