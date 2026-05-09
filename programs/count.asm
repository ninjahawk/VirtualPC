; count.asm — count 0 to 9, one number per line
; Uses address $80 as a loop counter variable
.org $00
    LDA #0
    STA $80       ; counter = 0

loop:
    LDA $80       ; load counter
    OUTN          ; print as decimal number
    LDA #$0A      ; newline
    OUT

    LDA $80       ; increment counter
    ADD #1
    STA $80

    CMP #10       ; if counter == 10, done
    JZ  done
    JMP loop

done:
    HLT
