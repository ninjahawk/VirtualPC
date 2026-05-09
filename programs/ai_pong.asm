; ai_pong.asm - Pong with neural-net AI right paddle
;
; Harvard architecture: code lives in CPU code store, data in file-backed memory.
; Variables never conflict with code regardless of program length.
;
; Network: 3 inputs -> 2 hidden (ReLU) -> 1 output
; Weights at $D0-$DA (written by run_ai_pong.py)
;
; Inputs (scaled >>2 to keep products inside signed 8-bit):
;   x0 = ball_y >> 2          (0-4)
;   x1 = (p2_y + 2) >> 2      (0-3)
;   x2 = 1 if dy==1, else 0
;
; Hidden i: pre_i = Wi0*x0 + Wi1*x1 + Wi2*x2 + bi
;           h_i   = ReLU(pre_i) >> 2
; Output:   out   = Vo0*h0 + Vo1*h1 + bo
;   out > 0  -> move AI paddle down
;   out < 0  -> move AI paddle up
;
; Weight layout at $D0:
;   $D0 W(by->h1)  $D1 W(pc->h1)  $D2 W(dy->h1)  $D3 b(h1)
;   $D4 W(by->h2)  $D5 W(pc->h2)  $D6 W(dy->h2)  $D7 b(h2)
;   $D8 W(h1->out) $D9 W(h2->out) $DA b(out)
;
; Scratch:  $E0 x0  $E1 x1  $E2 x2  $E3 h1  $E4 h2  $E5 acc
; Game:     $F0 bx  $F1 by  $F2 dx  $F3 dy
;           $F4 p1  $F5 p2  $F6 s1  $F7 s2  $F8 key

.org $00

    LDA #19
    STA $F0
    LDA #8
    STA $F1
    LDA #1
    STA $F2
    STA $F3
    LDA #6
    STA $F4
    STA $F5
    LDA #0
    STA $F6
    STA $F7
    DRAW

; ---- Main loop -------------------------------------------------------
loop:
    KEY
    STA $F8

    ; W/S: human controls LEFT paddle only
    CMP #119
    JNZ chk_s
    LDA $F4
    CMP #0
    JZ  chk_s
    SUB #1
    STA $F4
chk_s:
    LDA $F8
    CMP #115
    JNZ chk_q
    LDA $F4
    CMP #12
    JZ  chk_q
    ADD #1
    STA $F4
chk_q:
    LDA $F8
    CMP #113
    JZL done

; ---- Predict ball arrival y at x=38 --------------------------------
; $E6 = frames remaining = 38 - bx
    LDA #38
    SUB $F0
    STA $E6

    LDA $F3
    CMP #1
    JNZ pred_up

pred_dn:               ; dy=+1: arrival = by + frames
    LDA $F1
    ADD $E6
    STA $E6            ; $E6 = arrival
    CMP #17
    JC  pred_ok        ; < 17, done
    LDA #34            ; reflect off bottom: 34 - arrival
    SUB $E6
    STA $E6
    JMPL pred_ok

pred_up:               ; dy=-1: arrival = by - frames
    LDA $F1
    SUB $E6
    JNC pred_ok2       ; no borrow (by >= frames), result valid
    LDA $E6            ; borrow: negate -> frames - by
    SUB $F1
    STA $E6
    CMP #17
    JC  pred_ok        ; < 17, done
    LDA #34            ; reflect off bottom: 34 - arrival
    SUB $E6
    STA $E6
    JMPL pred_ok

pred_ok2:
    STA $E6
pred_ok:

; ---- Neural net inference -------------------------------------------
    ; x0 = arrival_y >> 1
    LDA $E6
    SHR
    STA $E0

    ; x1 = (p2_y + 2) >> 1
    LDA $F5
    ADD #2
    SHR
    STA $E1

    ; x2 = 1 if dy==1 else 0
    LDA $F3
    CMP #1
    JNZ dy_up
    LDA #1
    JMP dy_done
dy_up:
    LDA #0
dy_done:
    STA $E2

    ; hidden neuron 1
    LDA $E0
    MUL $D0
    STA $E5
    LDA $E1
    MUL $D1
    ADD $E5
    STA $E5
    LDA $E2
    MUL $D2
    ADD $E5
    ADD $D3
    JN  rl1
    JMP rl1d
rl1:
    LDA #0
rl1d:
    STA $E3

    ; hidden neuron 2
    LDA $E0
    MUL $D4
    STA $E5
    LDA $E1
    MUL $D5
    ADD $E5
    STA $E5
    LDA $E2
    MUL $D6
    ADD $E5
    ADD $D7
    JN  rl2
    JMP rl2d
rl2:
    LDA #0
rl2d:
    STA $E4

    ; output neuron
    LDA $E3
    MUL $D8
    STA $E5
    LDA $E4
    MUL $D9
    ADD $E5
    ADD $DA

    ; move right paddle on output sign
    JZL move_ball
    JN  ai_up
    LDA $F5
    CMP #12
    JZL move_ball
    ADD #1
    STA $F5
    JMP move_ball
ai_up:
    LDA $F5
    CMP #0
    JZL move_ball
    SUB #1
    STA $F5

; ---- Move ball -------------------------------------------------------
move_ball:
    LDA $F0
    ADD $F2
    STA $F0
    LDA $F1
    ADD $F3
    STA $F1

    ; bounce top/bottom
    LDA $F1
    CMP #0
    JZ  bncy
    CMP #17
    JNZ chkl
bncy:
    LDA $F3
    NOT
    INC
    STA $F3
    LDA $F1
    ADD $F3
    STA $F1

    ; left side
chkl:
    LDA $F0
    CMP #2
    JNCL chkr
    CMP #0
    JZL  scr2
    LDA $F1
    SUB $F4
    JCL  scr2
    CMP #5
    JNCL scr2
    LDA $F2
    NOT
    INC
    STA $F2
    LDA #2
    STA $F0
    JMPL redraw

    ; right side
chkr:
    LDA $F0
    CMP #38
    JCL  redraw
    JNZL scr1
    LDA $F1
    SUB $F5
    JCL  scr1
    CMP #5
    JNCL scr1
    LDA $F2
    NOT
    INC
    STA $F2
    LDA #37
    STA $F0
    JMPL redraw

scr1:
    LDA $F6
    ADD #1
    STA $F6
    LDA #1         ; serve toward AI (dx = +1)
    STA $F2
    JMPL rst_dy
scr2:
    LDA $F7
    ADD #1
    STA $F7
    LDA #255       ; serve toward human (dx = -1)
    STA $F2
rst_dy:
    LDA $F6        ; alternate dy each point: (s1+s2)&1
    ADD $F7
    AND #1
    JZL rst_dn
    LDA #255       ; dy = -1 (ball goes up)
    STA $F3
    JMPL rst_xy
rst_dn:
    LDA #1         ; dy = +1 (ball goes down)
    STA $F3
rst_xy:
    LDA #19
    STA $F0
    LDA #8
    STA $F1

redraw:
    DRAW
    WAIT
    JMP loop

done:
    HLT
