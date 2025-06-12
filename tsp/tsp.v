/*******************************************************************************
 * @file       tspFSM.v
 * @brief      用於 4 城市旅行推銷員問題（TSP）的有限狀態機（FSM）驗證器
 * @details    此模組遍歷所有以城市 0 為起點的路徑排列，計算總成本。
 *             若所有路徑成本都不低於定義的門檻（THRESHOLD），則輸出 property p=1；
 *             否則若找到更低成本路徑則 p=0，表示存在更短路徑。
 *             採同步時序設計，狀態更新與變數紀錄均依賴 clk 正緣。
 * 
 * @author     Jimmy Chang
 * @version    8.0
 * @date       2025-06-12
 * @license    MIT
 *
 * Circuit Statistics (present by GV)
 * ==================
   PI           2
   PO           1
   LATCH       18
   AIG        743
  ------------------
   Total      764

  * Conclusion
  * FSM-style TSP verifier for 4 cities; single property output p 
  * 如果都沒有 → foundLower = 0 → p = 1
  * | 編號 | 路徑（CITY 0 起點）     | 總成本計算                  |
  * | -- | ----------------- | ---------------------- |
  * | 1  | 0 → 1 → 2 → 3 → 0 | 3 + 5 + 6 + 2 = **16** |
  * | 2  | 0 → 1 → 3 → 2 → 0 | 3 + 1 + 6 + 4 = **14** |
  * | 3  | 0 → 2 → 1 → 3 → 0 | 4 + 5 + 1 + 2 = **12** |
  * | 4  | 0 → 2 → 3 → 1 → 0 | 4 + 6 + 1 + 3 = **14** |
  * | 5  | 0 → 3 → 1 → 2 → 0 | 2 + 1 + 5 + 4 = **12** |
  * | 6  | 0 → 3 → 2 → 1 → 0 | 2 + 6 + 5 + 3 = **16** |
  * 測試 p = 0 的情況，只需要降低 THRESHOLD 值，例如設成 THRESHOLD = 13，
  * 因為有成本 = 12 的路徑存在，foundLower 就會被設成 1，最後 p = 0。
 ******************************************************************************/

`define COST_0_1    8'd3
`define COST_0_2    8'd4
`define COST_0_3    8'd2
`define COST_1_0    8'd3
`define COST_1_2    8'd5
`define COST_1_3    8'd1
`define COST_2_0    8'd4
`define COST_2_1    8'd5
`define COST_2_3    8'd6
`define COST_3_0    8'd2
`define COST_3_1    8'd1
`define COST_3_2    8'd6
`define THRESHOLD   8'd12/ change this to test different thresholds

module tspFSM (
    output p,       // property output (1 if no tour cost < THRESHOLD)
    input  clk,
    input  reset
);

  // FSM state encoding
  reg [1:0] state, state_w;
  parameter S_INIT=2'd0, S_COMPUTE=2'd1, S_DONE=2'd2;

  // Tour indices (fix CITY_0 as start; permute CITY_1..CITY_3)
  reg [1:0] j, k, l;
  reg [1:0] j_w, k_w, l_w;

  // Cost tracking
  reg [7:0] minCost, minCost_w;
  reg       foundLower, foundLower_w;
  reg       initialized;

  // Compute pairwise distances as wires
  wire [7:0] cost0j = (j==2'b01) ? `COST_0_1 : (j==2'b10) ? `COST_0_2 : (j==2'b11) ? `COST_0_3 : 8'd0;
  wire [7:0] cost_jk = (j==2'b01 && k==2'b10) ? `COST_1_2 :
                       (j==2'b01 && k==2'b11) ? `COST_1_3 :
                       (j==2'b10 && k==2'b01) ? `COST_2_1 :
                       (j==2'b10 && k==2'b11) ? `COST_2_3 :
                       (j==2'b11 && k==2'b01) ? `COST_3_1 :
                       (j==2'b11 && k==2'b10) ? `COST_3_2 : 8'd0;
  wire [7:0] cost_kl = (k==2'b01 && l==2'b10) ? `COST_1_2 :
                       (k==2'b01 && l==2'b11) ? `COST_1_3 :
                       (k==2'b10 && l==2'b01) ? `COST_2_1 :
                       (k==2'b10 && l==2'b11) ? `COST_2_3 :
                       (k==2'b11 && l==2'b01) ? `COST_3_1 :
                       (k==2'b11 && l==2'b10) ? `COST_3_2 : 8'd0;
  wire [7:0] cost_l0 = (l==2'b01) ? `COST_1_0 :
                       (l==2'b10) ? `COST_2_0 :
                       (l==2'b11) ? `COST_3_0 : 8'd0;

  wire [7:0] totalCost = cost0j + cost_jk + cost_kl + cost_l0;

  // Property output: high only if done and no cost below threshold
  assign p = initialized && (state == S_DONE) && !foundLower;

  // Combinational next-state logic
  always @(*) begin
    // Default: hold all values
    state_w = state;
    j_w = j; k_w = k; l_w = l;
    minCost_w = minCost;
    foundLower_w = foundLower;

    case (state)
      S_INIT: begin
        // Initialize first tour to (CITY_0 -> CITY_1 -> CITY_2 -> CITY_3 -> CITY_0)
        state_w = S_COMPUTE;
        j_w = 2'd1; k_w = 2'd2; l_w = 2'd3;
        minCost_w = 8'd255;
        foundLower_w = 1'b0;
      end

      S_COMPUTE: begin
        // Update minimum cost and threshold flag
        if (totalCost < minCost)
          minCost_w = totalCost;
        if (totalCost < `THRESHOLD)
          foundLower_w = 1'b1;

        // Advance to next permutation:
        // 1) Try to increment l (keeping j,k constant)
        if (l < 2'd3) begin
          if ((l == 2'd1) && (2'd2!=j && 2'd2!=k))       l_w = 2'd2;
          else if ((l == 2'd1) && (2'd2==j || 2'd2==k) && (2'd3!=j && 2'd3!=k)) l_w = 2'd3;
          else if ((l == 2'd2) && (2'd3!=j && 2'd3!=k))  l_w = 2'd3;
        end
        // 2) If l did not advance, try to increment k
        if (l_w == l) begin
          if (k < 2'd3) begin
            if ((k == 2'd1) && (2'd2 != j))        k_w = 2'd2;
            else if ((k == 2'd1) && (2'd2 == j) && (2'd3 != j)) k_w = 2'd3;
            else if ((k == 2'd2) && (2'd3 != j))   k_w = 2'd3;
          end
          // If k changed, reset l to first valid value
          if (k_w != k) begin
            if (2'd1!=j && 2'd1!=k_w)        l_w = 2'd1;
            else if (2'd2!=j && 2'd2!=k_w)   l_w = 2'd2;
            else if (2'd3!=j && 2'd3!=k_w)   l_w = 2'd3;
          end
          // 3) If neither l nor k changed, try to increment j
          else begin
            if (j < 2'd3) begin
              j_w = j + 2'd1;
              // Set k to the lowest city not equal j_w
              if (j_w != 2'd1) k_w = 2'd1; else k_w = 2'd2;
              // Set l to the lowest city not equal j_w,k_w
              if (2'd1!=j_w && 2'd1!=k_w)      l_w = 2'd1;
              else if (2'd2!=j_w && 2'd2!=k_w) l_w = 2'd2;
              else if (2'd3!=j_w && 2'd3!=k_w) l_w = 2'd3;
            end else begin
              // All permutations done
              state_w = S_DONE;
            end
          end
        end
      end

      S_DONE: begin
        // Remain in DONE
        state_w = S_DONE;
      end
    endcase
  end

  // Sequential state-update
  always @(posedge clk) begin
    if (!reset) begin
      state      <= S_INIT;
      j          <= 2'd0;
      k          <= 2'd0;
      l          <= 2'd0;
      minCost    <= 8'd255;
      foundLower <= 1'b0;
      initialized <= 1'b0;
    end else begin
      state      <= state_w;
      j          <= j_w;
      k          <= k_w;
      l          <= l_w;
      minCost    <= minCost_w;
      foundLower <= foundLower_w;
      initialized <= 1'b1;
    end
  end

endmodule
