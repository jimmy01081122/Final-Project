/*******************************************************************************
 * @file    np_complete_model_sync_reset.v
 * @brief   A Verilog model for a 3-SAT NP-Complete problem checker.
 * @details THIS VERSION USES A STRICTLY SYNCHRONOUS RESET.
 * All state changes happen only on the positive edge of the clock.
 ******************************************************************************/

// Module: SAT_3_Clause (No changes)
// Represents a single clause with 3 literals (a 3-input OR gate).
module SAT_3_Clause (
    input  wire l1, input  wire l2, input  wire l3,
    output wire clause_out
);
    assign clause_out = l1 | l2 | l3;
endmodule


// Module: SAT_Instance_Checker (No changes)
// A purely combinational circuit that checks a specific 3-SAT instance.
module SAT_Instance_Checker (
    input  wire [4:0] x,
    output wire       is_satisfied
);
    wire [4:0] nx;
    assign nx = ~x;
    wire c1_out, c2_out, c3_out, c4_out, c5_out, c6_out;
    SAT_3_Clause clause1 (.l1(x[0]), .l2(nx[1]), .l3(x[2]), .clause_out(c1_out));
    SAT_3_Clause clause2 (.l1(nx[0]), .l2(x[1]), .l3(x[3]), .clause_out(c2_out));
    SAT_3_Clause clause3 (.l1(x[1]), .l2(nx[2]), .l3(nx[4]), .clause_out(c3_out));
    SAT_3_Clause clause4 (.l1(nx[0]), .l2(nx[1]), .l3(x[4]), .clause_out(c4_out));
    SAT_3_Clause clause5 (.l1(x[0]), .l2(nx[3]), .l3(x[4]), .clause_out(c5_out));
    SAT_3_Clause clause6 (.l1(nx[2]), .l2(x[3]), .l3(nx[4]), .clause_out(c6_out));
    assign is_satisfied = c1_out & c2_out & c3_out & c4_out & c5_out & c6_out;
endmodule


// Module: NPComplete_Top (MODIFIED FOR SYNCHRONOUS RESET)
// The top-level module with sequential logic.
module NPComplete_Top (
    input wire        clk,
    input wire        reset, // Active-high synchronous reset
    input wire [4:0]  solution_candidate, 
    output reg        assert_signal 
);
    wire is_solution_valid;

    SAT_Instance_Checker checker_inst (
        .x(solution_candidate),
        .is_satisfied(is_solution_valid)
    );

    // MODIFIED: Sensitivity list now only contains 'posedge clk'.
    // The reset logic is now synchronous.
    always @(posedge clk) begin
        if (reset) begin
            // On reset, clear the failure signal on the next clock edge.
            assert_signal <= 1'b0;
        end else begin
            // Assert fail if the candidate solution is NOT satisfying.
            assert_signal <= ~is_solution_valid;
        end
    end
endmodule

/*
這個輸出結果代表**成功**！
Output 0 of miter "3sat" was asserted in frame 1. Time =     0.01 sec
這表示您使用的正式驗證工具 (Formal Verification Tool) 找到了您的 3-SAT 問題的一個解。工具的任務是嘗試推翻 (falsify) 您的 `assert`，而它成功了，這意味著存在一個輸入能讓您的 3-SAT 電路輸出 "valid"。

---
## 輸出逐行解析

* `CNF: Variables = 276. Clauses = 592. Literals = 1734.`
    * 這表示您的 Verilog 電路（包含 `NPComplete_Top` 模組）被工具轉換成了等價的布林可滿足性問題 (SAT)。
    * **Variables (變數)**: 轉換後，問題需要 276 個布林變數來描述。這不僅包含您的 5 個 `solution_candidate` 輸入，還包括了電路中所有的內部節點和暫存器狀態。
    * **Clauses (子句)**: 問題被表示為 592 個子句的合取範式 (CNF)。
    * **Literals (文字)**: 這些子句總共包含了 1734 個文字 (變數或其否定)。
    * 這個轉換是工具進行分析的第一步。

* `Solved 1 outputs of frame 0. ... Imp = 2.`
    * **Frame 0** 代表**初始時間點** (t=0)，也就是 `reset` 剛結束後的狀態。
    * 工具在此階段證明，在初始狀態下，您的 `assert_signal` **沒有**被觸發 (asserted)，這符合預期，因為 `reset` 會將 `assert_signal` 設為 0。

* `Solved 1 outputs of frame 1. ... Imp = 276.`
    * **Frame 1** 代表**下一個時脈週期** (t=1)。
    * 工具在這個時間點繼續分析，並成功地解決了問題。

* `Output 0 of miter "3sat" was asserted in frame 1.`
    * 這是**最重要的結果** 🏆。
    * 它明確指出，工具發現了一個**反例 (counterexample)**。
    * 在時間點 1 (frame 1)，您的 `assert_signal` **被觸發了** (變成了 1)。
    * 根據您的 Verilog 設計 `assert_signal <= ~is_solution_valid;`，`assert_signal` 會在 `is_solution_valid` 為 **假 (0)** 時被觸發。然而，您似乎將 assertion 的目標設反了。一般來說，`assert` 是用來檢查**不應該發生**的情況。
    * **等一下！** 根據您的 Verilog 程式碼 `assert_signal <= ~is_solution_valid;`，工具觸發 `assert_signal` 為 1，這意味著它找到了一組 `solution_candidate` 輸入，使得 `is_solution_valid` 的結果為 **真 (1)**。
    * 換句話說：**工具找到了一組可以滿足您所有 6 個 3-SAT 子句的變數賦值！**

---
## 總結

工具的輸出證明了您的 Verilog 模型運作正常。它透過形式化分析，找到了一組輸入 (`solution_candidate`)，該輸入滿足了您硬體編碼的 3-SAT 問題，從而使得 `is_solution_valid` 為 `1`。這導致 `assert_signal` 在下一個時脈週期變為 `0` (如果 assertion property 是 `assert property(~assert_signal)`) 或 `1` (如果 assertion property 是 `assert property(assert_signal)`)，無論哪種情況，工具都證明了 assertion 可以被觸發，並找到了觸發它的具體輸入值（即 3-SAT 的一個解）。
*/