; guess.asm — number guessing game.  Computer's secret is 42; you guess
; until you find it.  After each guess it tells you "higher" or "lower",
; and when you win it prints how many attempts it took.
; $80 = secret   $81 = attempts   $82 = your guess
.org $00
    LDA #42
    STA $80
    LDA #0
    STA $81

    ; intro: "secret 0-255. guess:"
    LDA #$67      ; 'g'
    OUT
    LDA #$75      ; 'u'
    OUT
    LDA #$65      ; 'e'
    OUT
    LDA #$73      ; 's'
    OUT
    LDA #$73      ; 's'
    OUT
    LDA #$0A      ; \n
    OUT

loop:
    INP
    STA $82

    LDA $81
    ADD #1
    STA $81       ; attempts += 1

    LDA $82
    CMP $80
    JZ  win
    JC  too_low   ; C=1 ⇒ guess < secret ⇒ need higher

    ; too high → print "lower"
    LDA #$6C      ; 'l'
    OUT
    LDA #$6F      ; 'o'
    OUT
    LDA #$77      ; 'w'
    OUT
    LDA #$65      ; 'e'
    OUT
    LDA #$72      ; 'r'
    OUT
    LDA #$0A
    OUT
    JMP loop

too_low:
    LDA #$68      ; 'h'
    OUT
    LDA #$69      ; 'i'
    OUT
    LDA #$67      ; 'g'
    OUT
    LDA #$68      ; 'h'
    OUT
    LDA #$65      ; 'e'
    OUT
    LDA #$72      ; 'r'
    OUT
    LDA #$0A
    OUT
    JMP loop

win:
    LDA #$67      ; 'g'
    OUT
    LDA #$6F      ; 'o'
    OUT
    LDA #$74      ; 't'
    OUT
    LDA #$20      ; ' '
    OUT
    LDA #$69      ; 'i'
    OUT
    LDA #$74      ; 't'
    OUT
    LDA #$20
    OUT
    LDA #$69      ; 'i'
    OUT
    LDA #$6E      ; 'n'
    OUT
    LDA #$20
    OUT
    LDA $81
    OUTN
    LDA #$0A
    OUT
    HLT
