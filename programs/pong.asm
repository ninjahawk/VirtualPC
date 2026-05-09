; pong.asm — Pong for VirtualPC 8-bit
;
; Controls:  W / S = left player up / down
;            O / L = right player up / down
;            Q     = quit
;
; Variables live at $F0-$F8 (above program code, below stack at $EF)
;   $F0  ball_x     (active range 1–38)
;   $F1  ball_y     (0–17)
;   $F2  ball_dx    (1=right, $FF=left)
;   $F3  ball_dy    (1=down,  $FF=up)
;   $F4  p1_y       (left paddle top,  0–12)
;   $F5  p2_y       (right paddle top, 0–12)
;   $F6  score1     (left)
;   $F7  score2     (right)
;   $F8  key        (scratch)

.org $00

; ── Init ────────────────────────────────────────────────────────────────────
    LDA #19
    STA $F0        ; ball_x = 19
    LDA #8
    STA $F1        ; ball_y = 8
    LDA #1
    STA $F2        ; dx = +1
    STA $F3        ; dy = +1
    LDA #6
    STA $F4        ; p1_y = 6
    STA $F5        ; p2_y = 6
    LDA #0
    STA $F6        ; score1 = 0
    STA $F7        ; score2 = 0
    DRAW

; ── Main loop ───────────────────────────────────────────────────────────────
loop:
    KEY
    STA $F8

    ; W: left player up
    CMP #119
    JNZ chk_s
    LDA $F4
    CMP #0
    JZ  chk_s
    SUB #1
    STA $F4
chk_s:
    LDA $F8
    CMP #115       ; S: left player down
    JNZ chk_o
    LDA $F4
    CMP #12
    JZ  chk_o
    ADD #1
    STA $F4
chk_o:
    LDA $F8
    CMP #111       ; O: right player up
    JNZ chk_l
    LDA $F5
    CMP #0
    JZ  chk_l
    SUB #1
    STA $F5
chk_l:
    LDA $F8
    CMP #108       ; L: right player down
    JNZ chk_q
    LDA $F5
    CMP #12
    JZ  chk_q
    ADD #1
    STA $F5
chk_q:
    LDA $F8
    CMP #113       ; Q: quit
    JZ  done

; ── Move ball ───────────────────────────────────────────────────────────────
    LDA $F0
    ADD $F2
    STA $F0        ; ball_x += dx

    LDA $F1
    ADD $F3
    STA $F1        ; ball_y += dy

; ── Top / bottom bounce ─────────────────────────────────────────────────────
    LDA $F1
    CMP #0
    JZ  bncy
    CMP #17
    JNZ chkl
bncy:
    LDA $F3
    NOT
    INC
    STA $F3        ; dy = -dy
    LDA $F1
    ADD $F3
    STA $F1        ; nudge ball one step in new direction

; ── Left side ───────────────────────────────────────────────────────────────
chkl:
    LDA $F0
    CMP #2
    JNC chkr       ; ball_x >= 2 → check right
    CMP #0
    JZ  scr2       ; ball_x == 0 → P2 scores
    ; ball_x == 1: test left paddle
    LDA $F1
    SUB $F4        ; ball_y - p1_y
    JC  scr2       ; above paddle
    CMP #5
    JNC scr2       ; below paddle
    ; HIT
    LDA $F2
    NOT
    INC
    STA $F2
    LDA #2
    STA $F0
    JMP redraw

; ── Right side ──────────────────────────────────────────────────────────────
chkr:
    LDA $F0
    CMP #38
    JC  redraw     ; ball_x < 38
    JNZ scr1       ; ball_x > 38 → P1 scores
    ; ball_x == 38: test right paddle
    LDA $F1
    SUB $F5
    JC  scr1
    CMP #5
    JNC scr1
    ; HIT
    LDA $F2
    NOT
    INC
    STA $F2
    LDA #37
    STA $F0
    JMP redraw

; ── Scoring ─────────────────────────────────────────────────────────────────
scr1:
    LDA $F6
    ADD #1
    STA $F6
    JMP rst
scr2:
    LDA $F7
    ADD #1
    STA $F7
rst:
    LDA #19
    STA $F0
    LDA #8
    STA $F1
    LDA #1
    STA $F2
    STA $F3

; ── Draw & next frame ───────────────────────────────────────────────────────
redraw:
    DRAW
    WAIT
    JMP loop

done:
    HLT
