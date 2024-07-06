//############################################################################
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//   (C) Copyright Laboratory System Integration and Silicon Implementation
//   All Right Reserved
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//   ICLAB 2023 Fall
//   Lab04 Exercise		: Convolution Neural Network 
//   Author     		: Cheng-Te Chang (chengdez.ee12@nycu.edu.tw)
//
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//   File Name   : CNN.v
//   Module Name : CNN
//   Release version : V1.0 (Release Date: 2024-02)
//
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//############################################################################

module CNN(
    //Input Port
    clk,
    rst_n,
    in_valid,
    Img,
    Kernel,
	Weight,
    Opt,

    //Output Port
    out_valid,
    out
    );


//---------------------------------------------------------------------
//   PARAMETER
//---------------------------------------------------------------------

// IEEE floating point parameter
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
parameter inst_arch_type = 0;
parameter inst_arch = 0;
parameter inst_faithful_round = 0;

input rst_n, clk, in_valid;
input [inst_sig_width+inst_exp_width:0] Img, Kernel, Weight;
input [1:0] Opt;

output reg	out_valid;
output reg [inst_sig_width+inst_exp_width:0] out;


wire [31:0] pixel_out0;
wire [31:0] pixel_out1;
wire [31:0] pixel_out2;
wire [31:0] nxt_out0;
wire [31:0] nxt_out1;
wire [31:0] nxt_out2;
wire ready;


reg [6:0] input_cnt;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) input_cnt <= 7'd0;
    else begin
       if (in_valid) input_cnt <= input_cnt + 1'b1;
       else          input_cnt <= 7'd0;
    end
end


reg [1:0] s_opt; 
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) s_opt <= 2'b0;
    else begin
        if (in_valid && input_cnt == 0) s_opt <= Opt;
        else                            s_opt <= s_opt;
    end
end
 

out_pixel pix1(
    .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .Opt_0(s_opt[1]), .ready(ready), .reset_sig(out_valid), .pixel_in(Img),
    .pixel_out0(pixel_out0),
    .pixel_out1(pixel_out1), .pixel_out2(pixel_out2),
    .nxt_out0(nxt_out0),
    .nxt_out1(nxt_out1), .nxt_out2(nxt_out2)
);

wire [31:0] k00, k01, k02;
wire [31:0] k10, k11, k12;
wire [31:0] k20, k21, k22;
wire [31:0] nxt_k00, nxt_k01;
wire [31:0] nxt_k10, nxt_k11;
wire [31:0] nxt_k20, nxt_k21;

out_kernel k1(
    .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .reset_sig(out_valid), .kernel_in(Kernel),
    .nxt_k00(nxt_k00), .nxt_k01(nxt_k01),
    .nxt_k10(nxt_k10), .nxt_k11(nxt_k11),
    .nxt_k20(nxt_k20), .nxt_k21(nxt_k21), .k00(k00), .k01(k01), .k02(k02),
    .k10(k10), .k11(k11), .k12(k12),
    .k20(k20), .k21(k21), .k22(k22)
);

wire [31:0] sum;
out_pearray pa1(
    .clk(clk), .rst_n(rst_n), .PE_in_en(ready),
    .k00(k00), .k01(k01), .k02(k02),
    .k10(k10), .k11(k11), .k12(k12),
    .k20(k20), .k21(k21), .k22(k22),
    .nxt_k00(nxt_k00), .nxt_k01(nxt_k01), .nxt_k10(nxt_k10), .nxt_k11(nxt_k11),
    .nxt_k20(nxt_k20), .nxt_k21(nxt_k21),
    .pixel_in0(pixel_out0), .pixel_in1(pixel_out1), .pixel_in2(pixel_out2),
    .nxt_in0(nxt_out0), .nxt_in1(nxt_out1), .nxt_in2(nxt_out2),
    .sum(sum)
);

wire [31:0] conv_out;
wire [5:0] feature_cnt;
wire pooling_ready;
out_convolution cr1(
    .clk(clk), .rst_n(rst_n), .ready(ready), .reset_signal(out_valid), .sum(sum), .conv_out(conv_out), .feature_cnt(feature_cnt)
);



wire [31:0] out_pooling;
Max_pooling mp1(
    .clk(clk), .rst_n(rst_n), .feature_cnt(feature_cnt), .conv_out(conv_out), .reset_signal(out_valid),
    .out_pooling(out_pooling), .pooling_ready(pooling_ready)
);

wire [31:0] out_fc;
wire [3:0] fc_cnt;
FC_normal fc1(
    .clk(clk), .rst_n(rst_n), .weight(Weight), .out_pooling(out_pooling), .pooling_ready(pooling_ready), .input_cnt(input_cnt), 
    .in_valid(in_valid), .reset_signal(out_valid),
    .out_fc(out_fc), .fc_cnt(fc_cnt)
);

wire [31:0] biggest, smallest, min_max_output;
wire norm_ready;
Normalization N1(
    .clk(clk), .rst_n(rst_n), .reset_signal(in_valid), .out_fc(out_fc), .fc_cnt(fc_cnt), 
    .biggest(biggest), .smallest(smallest), .min_max_output(min_max_output), .norm_ready(norm_ready)
);



wire [31:0] act_out;
wire act_ready;
Activation A1(
    .clk(clk), .rst_n(rst_n), .biggest(biggest), .smallest(smallest), .min_max_output(min_max_output), .norm_ready(norm_ready), .s_opt(s_opt), .fc_cnt(fc_cnt),
    .act_ready(act_ready), .act_out(act_out)
);


always @ (posedge clk, negedge rst_n) begin
    if(!rst_n) begin
        out_valid <= 0;
        out <= 0;
    end
    else begin
        if(act_ready && !in_valid) begin
            out_valid <= 1;
            out <= act_out;
        end
        else begin
            out_valid <= 0;
            out <= 0;
        end
    end
end


endmodule

module out_pixel(
    clk, rst_n, in_valid, Opt_0, ready, reset_sig,
    pixel_in,
    pixel_out0,
    pixel_out1,
    pixel_out2,
    nxt_out0,
    nxt_out1,
    nxt_out2
);
input  clk, rst_n, in_valid, Opt_0, reset_sig; // Opt_0 = 1'b0 -> Replication Padding, Opt_0 = 1'b1 -> Zero Padding 
input  [31:0] pixel_in;
output reg [31:0] pixel_out0, pixel_out1, pixel_out2;
output reg [31:0] nxt_out0, nxt_out1, nxt_out2;
output reg ready;

reg [31:0] p11, p12, p13, p14;
reg [31:0] p21, p22, p23, p24;
reg [31:0] p31, p32, p33, p34;
reg [31:0] p41, p42, p43, p44;

reg [31:0] Corner0, Corner1, Corner2, Corner3; 
reg [31:0] Up0, Up1; 
reg [31:0] Down0, Down1; 
reg [31:0] Left0, Left1; 
reg [31:0] Right0, Right1; 

reg [3:0] in_cnt;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) in_cnt <= 4'd0;
    else begin
        if (in_valid) in_cnt <= in_cnt + 1'b1; 
        else          in_cnt <= 4'd0;
    end
end

always @* begin
    if (!Opt_0) begin
        Corner0 = 32'd0; Corner1 = 32'd0; Corner2 = 32'd0; Corner3 = 32'd0;
        Up0 = 32'd0; Up1 = 32'd0; Down0 = 32'd0; Down1 = 32'd0;
        Left0 = 32'd0; Left1 = 32'd0; Right0 = 32'd0; Right1 = 32'd0;
    end
    else begin
        Corner0 = p11; Corner1 = p14; Corner2 = p41; Corner3 = p44;
        Up0 = p12; Up1 = p13; Down0 = p42; Down1 = p43;
        Left0 = p21; Left1 = p31; Right0 = p24; Right1 = p34;
    end
end


reg [4:0] out_cnt;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        out_cnt <= 5'd0;
        ready   <= 1'b0;
    end
    else begin
        if (reset_sig) begin
            out_cnt <= 5'd0;
            ready   <= 1'b0;
        end
        else begin
            if (in_cnt < 4'd6 && !ready) begin // can start output data after in_cnt > 5
                out_cnt <= out_cnt;
                ready   <= ready;
            end
            else begin
                ready   <= 1'b1;
                if (out_cnt == 5'd18) out_cnt <= 5'd3;
                else                  out_cnt <= out_cnt + 1;
            end
        end
    end
end

always @* begin
    case (out_cnt)
        5'd1: begin
            pixel_out0 = Corner0; nxt_out0 = Up0;
            pixel_out1 = Corner0; nxt_out1 = p12;
            pixel_out2 = Left0; nxt_out2 = p22;
        end
        5'd2: begin
            pixel_out0 = Corner0;  nxt_out0 = Up1; 
            pixel_out1 = p11; nxt_out1 = p13;
            pixel_out2 = p21; nxt_out2 = p23;  
        end
        5'd3: begin
            pixel_out0 = Up0;  nxt_out0 = Corner1; 
            pixel_out1 = p12; nxt_out1 = p14;
            pixel_out2 = p22; nxt_out2 = p24;  
        end
        5'd4: begin
            pixel_out0 = Up1;  nxt_out0 = Corner1; 
            pixel_out1 = p13; nxt_out1 = Corner1;
            pixel_out2 = p23; nxt_out2 = Right0;  
        end
        5'd5: begin
            pixel_out0 = Corner1;  nxt_out0 = Corner0; 
            pixel_out1 = p14; nxt_out1 = Left0;
            pixel_out2 = p24; nxt_out2 = Left1;  
        end
        5'd6: begin
            pixel_out0 = Corner1; nxt_out0 = p11; 
            pixel_out1 = Corner1; nxt_out1 = p21;
            pixel_out2 = Right0; nxt_out2 = p31;  
        end
        5'd7: begin
            pixel_out0 = p12; nxt_out0 = p14; 
            pixel_out1 = p22; nxt_out1 = p24;
            pixel_out2 = p32; nxt_out2 = p34;  
        end
        5'd8: begin
            pixel_out0 = p13; nxt_out0 = Corner1; 
            pixel_out1 = p23; nxt_out1 = Right0;
            pixel_out2 = p33; nxt_out2 = Right1;  
        end
        5'd9: begin
            pixel_out0 = p14; nxt_out0 = Left0; 
            pixel_out1 = p24; nxt_out1 = Left1;
            pixel_out2 = p34; nxt_out2 = Corner2;  
        end
        5'd10: begin
            pixel_out0 = Corner1; nxt_out0 = p21; 
            pixel_out1 = Right0; nxt_out1 = p31;
            pixel_out2 = Right1; nxt_out2 = p41;  
        end
        5'd11: begin
            pixel_out0 = p22; nxt_out0 = p24; 
            pixel_out1 = p32; nxt_out1 = p34;
            pixel_out2 = p42; nxt_out2 = p44;  
        end
        5'd12: begin
            pixel_out0 = p23; nxt_out0 = Right0; 
            pixel_out1 = p33; nxt_out1 = Right1;
            pixel_out2 = p43; nxt_out2 = Corner3;  
        end
        5'd13: begin
            pixel_out0 = p24; nxt_out0 = Left1; 
            pixel_out1 = p34; nxt_out1 = Corner2;
            pixel_out2 = p44; nxt_out2 = Corner2;  
        end
        5'd14: begin
            pixel_out0 = Right0; nxt_out0 = p31; 
            pixel_out1 = Right1; nxt_out1 = p41;
            pixel_out2 = Corner3; nxt_out2 = Corner2;  
        end
        5'd15: begin
            pixel_out0 = p32; nxt_out0 = p34; 
            pixel_out1 = p42; nxt_out1 = p44;
            pixel_out2 = Down0;  nxt_out2 = Corner3;  
        end 
        5'd16: begin
            pixel_out0 = p33; nxt_out0 = Right1; 
            pixel_out1 = p43; nxt_out1 = Corner3;
            pixel_out2 = Down1;  nxt_out2 = Corner3;  
        end
        5'd17: begin
            pixel_out0 = p34; nxt_out0 = Corner0; 
            pixel_out1 = p44; nxt_out1 = Corner0;
            pixel_out2 = Corner3;  nxt_out2 = Left0;  
        end
        5'd18: begin
            pixel_out0 = Right1; nxt_out0 = Corner0; 
            pixel_out1 = Corner3; nxt_out1 = p11;
            pixel_out2 = Corner3; nxt_out2 = p21;  
        end
        default: begin
            pixel_out0 = 32'd0; nxt_out0 = 32'd0;
            pixel_out1 = 32'd0; nxt_out1 = 32'd0;
            pixel_out2 = 32'd0; nxt_out2 = 32'd0;  
        end
    endcase
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        p11 <= 32'd0;  p12 <= 32'd0;  p13 <= 32'd0;  p14 <= 32'd0;
        p21 <= 32'd0;  p22 <= 32'd0;  p23 <= 32'd0;  p24 <= 32'd0;
        p31 <= 32'd0;  p32 <= 32'd0;  p33 <= 32'd0;  p34 <= 32'd0;
        p41 <= 32'd0;  p42 <= 32'd0;  p43 <= 32'd0;  p44 <= 32'd0;
    end
    else begin
        if (in_valid) begin
            case (in_cnt)
                4'd0 : p11 <= pixel_in;
                4'd1 : p12 <= pixel_in;
                4'd2 : p13 <= pixel_in;
                4'd3 : p14 <= pixel_in;
                4'd4 : p21 <= pixel_in;
                4'd5 : p22 <= pixel_in;
                4'd6 : p23 <= pixel_in;
                4'd7 : p24 <= pixel_in;
                4'd8 : p31 <= pixel_in;
                4'd9 : p32 <= pixel_in;
                4'd10: p33 <= pixel_in;
                4'd11: p34 <= pixel_in;
                4'd12: p41 <= pixel_in;
                4'd13: p42 <= pixel_in;
                4'd14: p43 <= pixel_in;
                4'd15: p44 <= pixel_in;
            endcase
        end
        else begin
            p11 <= p11;  p12 <= p12;  p13 <= p13;  p14 <= p14;
            p21 <= p21;  p22 <= p22;  p23 <= p23;  p24 <= p24;
            p31 <= p31;  p32 <= p32;  p33 <= p33;  p34 <= p34;
            p41 <= p41;  p42 <= p42;  p43 <= p43;  p44 <= p44;
        end
    end
end




endmodule

module out_kernel(
    clk, rst_n, in_valid, reset_sig,
    kernel_in,
    k00, k01, k02,
    k10, k11, k12,
    k20, k21, k22,
    nxt_k00, nxt_k01,
    nxt_k10, nxt_k11,
    nxt_k20, nxt_k21
);
input  clk, rst_n, in_valid, reset_sig; // Opt_0 = 1'b0 -> Replication Padding, Opt_0 = 1'b1 -> Zero Padding 
input  [31:0] kernel_in;
output reg [31:0] k00, k01, k02;
output reg [31:0] k10, k11, k12;
output reg [31:0] k20, k21, k22;
output reg [31:0] nxt_k00, nxt_k01;
output reg [31:0] nxt_k10, nxt_k11;
output reg [31:0] nxt_k20, nxt_k21;

// kernel
reg [31:0] k00_0, k01_0, k02_0;
reg [31:0] k10_0, k11_0, k12_0;
reg [31:0] k20_0, k21_0, k22_0;

reg [31:0] k00_1, k01_1, k02_1;
reg [31:0] k10_1, k11_1, k12_1;
reg [31:0] k20_1, k21_1, k22_1;

reg [31:0] k00_2, k01_2, k02_2;
reg [31:0] k10_2, k11_2, k12_2;
reg [31:0] k20_2, k21_2, k22_2;


reg [4:0] in_cnt;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) in_cnt <= 5'd0;
    else begin
        if (in_valid) begin 
            if (in_cnt < 5'd27) in_cnt <= in_cnt + 1'b1;
            else                in_cnt <= in_cnt;
        end 
        else          in_cnt <= 5'd0;
    end
end

reg [5:0] out_cnt;
reg ready;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        out_cnt <= 6'd0;
        ready   <= 1'b0;
    end
    else begin
        if (reset_sig) begin
            out_cnt <= 6'd0;
            ready   <= 1'b0;
        end
        else begin
            if (in_cnt < 4'd6 && !ready) begin // can start output data after in_cnt > 5
                out_cnt <= out_cnt;
                ready   <= ready;
            end
            else begin
                ready   <= 1'b1;
                if (out_cnt == 6'd50) out_cnt <= 6'd3;
                else                  out_cnt <= out_cnt + 1;
            end
        end 
    end
end

always @* begin
    case (out_cnt)
        6'd1, 6'd2, 6'd3, 6'd4, 6'd5, 6'd6, 6'd7, 6'd8, 6'd9, 6'd10,
        6'd11, 6'd12, 6'd13, 6'd14, 6'd15, 6'd16, 6'd17, 6'd18: begin
            k00 = k00_0; k01 = k01_0; k02 = k02_0;
            k10 = k10_0; k11 = k11_0; k12 = k12_0;
            k20 = k20_0; k21 = k21_0; k22 = k22_0;
            nxt_k00 = k00_1; nxt_k01 = k01_1;
            nxt_k10 = k10_1; nxt_k11 = k11_1;
            nxt_k20 = k20_1; nxt_k21 = k21_1;
        end
        6'd19, 6'd20, 6'd21, 6'd22, 6'd23, 6'd24, 6'd25, 6'd26, 6'd27, 6'd28, 
        6'd29, 6'd30, 6'd31, 6'd32, 6'd33, 6'd34: begin
            k00 = k00_1; k01 = k01_1; k02 = k02_1;
            k10 = k10_1; k11 = k11_1; k12 = k12_1;
            k20 = k20_1; k21 = k21_1; k22 = k22_1;
            nxt_k00 = k00_2; nxt_k01 = k01_2;
            nxt_k10 = k10_2; nxt_k11 = k11_2;
            nxt_k20 = k20_2; nxt_k21 = k21_2;
        end
        6'd35, 6'd36, 6'd37, 6'd38, 6'd39, 6'd40, 6'd41, 6'd42, 6'd43, 6'd44,
        6'd45, 6'd46, 6'd47, 6'd48, 6'd49, 6'd50: begin
            k00 = k00_2; k01 = k01_2; k02 = k02_2;
            k10 = k10_2; k11 = k11_2; k12 = k12_2;
            k20 = k20_2; k21 = k21_2; k22 = k22_2;
            nxt_k00 = k00_0; nxt_k01 = k01_0;
            nxt_k10 = k10_0; nxt_k11 = k11_0;
            nxt_k20 = k20_0; nxt_k21 = k21_0;
        end
        default: begin
            k00 = 32'd0; k01 = 32'd0; k02 = 32'd0;
            k10 = 32'd0; k11 = 32'd0; k12 = 32'd0;
            k20 = 32'd0; k21 = 32'd0; k22 = 32'd0;
            nxt_k00 = 32'd0; nxt_k01 = 32'd0;
            nxt_k10 = 32'd0; nxt_k11 = 32'd0;
            nxt_k20 = 32'd0; nxt_k21 = 32'd0;
        end
    endcase
end


always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        k00_0 <= 32'd0; k01_0 <= 32'd0; k02_0 <= 32'd0;
        k10_0 <= 32'd0; k11_0 <= 32'd0; k12_0 <= 32'd0;
        k20_0 <= 32'd0; k21_0 <= 32'd0; k22_0 <= 32'd0;
        
        k00_1 <= 32'd0; k01_1 <= 32'd0; k02_1 <= 32'd0;
        k10_1 <= 32'd0; k11_1 <= 32'd0; k12_1 <= 32'd0;
        k20_1 <= 32'd0; k21_1 <= 32'd0; k22_1 <= 32'd0;
        
        k00_2 <= 32'd0; k01_2 <= 32'd0; k02_2 <= 32'd0;
        k10_2 <= 32'd0; k11_2 <= 32'd0; k12_2 <= 32'd0;
        k20_2 <= 32'd0; k21_2 <= 32'd0; k22_2 <= 32'd0;
    end
    else begin
        if (in_valid) begin
            case (in_cnt)
                5'd0 : k00_0 <= kernel_in;
                5'd1 : k01_0 <= kernel_in;
                5'd2 : k02_0 <= kernel_in;
                5'd3 : k10_0 <= kernel_in;
                5'd4 : k11_0 <= kernel_in;
                5'd5 : k12_0 <= kernel_in;
                5'd6 : k20_0 <= kernel_in;
                5'd7 : k21_0 <= kernel_in;
                5'd8 : k22_0 <= kernel_in;
                5'd9 : k00_1 <= kernel_in;
                5'd10: k01_1 <= kernel_in;
                5'd11: k02_1 <= kernel_in;
                5'd12: k10_1 <= kernel_in;
                5'd13: k11_1 <= kernel_in;
                5'd14: k12_1 <= kernel_in;
                5'd15: k20_1 <= kernel_in;
                5'd16: k21_1 <= kernel_in;
                5'd17: k22_1 <= kernel_in;
                5'd18: k00_2 <= kernel_in;
                5'd19: k01_2 <= kernel_in;
                5'd20: k02_2 <= kernel_in;
                5'd21: k10_2 <= kernel_in;
                5'd22: k11_2 <= kernel_in;
                5'd23: k12_2 <= kernel_in;
                5'd24: k20_2 <= kernel_in;
                5'd25: k21_2 <= kernel_in;
                5'd26: k22_2 <= kernel_in;
                default: begin
                    k00_0 <= k00_0; k01_0 <= k01_0; k02_0 <= k02_0;
                    k10_0 <= k10_0; k11_0 <= k11_0; k12_0 <= k12_0;
                    k20_0 <= k20_0; k21_0 <= k21_0; k22_0 <= k22_0;
                    
                    k00_1 <= k00_1; k01_1 <= k01_1; k02_1 <= k02_1;
                    k10_1 <= k10_1; k11_1 <= k11_1; k12_1 <= k12_1;
                    k20_1 <= k20_1; k21_1 <= k21_1; k22_1 <= k22_1;
                    
                    k00_2 <= k00_2; k01_2 <= k01_2; k02_2 <= k02_2;
                    k10_2 <= k10_2; k11_2 <= k11_2; k12_2 <= k12_2;
                    k20_2 <= k20_2; k21_2 <= k21_2; k22_2 <= k22_2;
                end
            endcase
        end
        else begin
            k00_0 <= k00_0; k01_0 <= k01_0; k02_0 <= k02_0;
            k10_0 <= k10_0; k11_0 <= k11_0; k12_0 <= k12_0;
            k20_0 <= k20_0; k21_0 <= k21_0; k22_0 <= k22_0;
            
            k00_1 <= k00_1; k01_1 <= k01_1; k02_1 <= k02_1;
            k10_1 <= k10_1; k11_1 <= k11_1; k12_1 <= k12_1;
            k20_1 <= k20_1; k21_1 <= k21_1; k22_1 <= k22_1;
            
            k00_2 <= k00_2; k01_2 <= k01_2; k02_2 <= k02_2;
            k10_2 <= k10_2; k11_2 <= k11_2; k12_2 <= k12_2;
            k20_2 <= k20_2; k21_2 <= k21_2; k22_2 <= k22_2;
        end
    end
end



endmodule


module out_pearray(
    clk, rst_n, PE_in_en,
    k00, k01, k02,
    k10, k11, k12,
    k20, k21, k22,
    nxt_k00, nxt_k01,
    nxt_k10, nxt_k11,
    nxt_k20, nxt_k21,
    pixel_in0, pixel_in1, pixel_in2,
    nxt_in0, nxt_in1, nxt_in2,
    sum
);
input clk, rst_n, PE_in_en;
input [31:0] k00, k01, k02;
input [31:0] k10, k11, k12;
input [31:0] k20, k21, k22;
input [31:0] nxt_k00, nxt_k01;
input [31:0] nxt_k10, nxt_k11;
input [31:0] nxt_k20, nxt_k21;
input [31:0] pixel_in0, pixel_in1, pixel_in2;
input [31:0] nxt_in0, nxt_in1, nxt_in2;
output [31:0] sum;

reg [31:0] kernel_00, kernel_01;
reg [31:0] kernel_10, kernel_11;
reg [31:0] kernel_20, kernel_21;


wire [31:0] bias_L1 [0:2], bias_L2 [0:2], bias_L3 [0:2];
reg [31:0] p00, p01, p02;
reg [31:0] p10, p11, p12;
reg [31:0] p20, p21, p22;



reg [4:0] pe_cnt;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) pe_cnt <= 5'd0;
    else begin
        if (PE_in_en) begin
            if (pe_cnt == 5'd17) pe_cnt <= 5'd2;
            else                 pe_cnt <= pe_cnt + 1;
        end
        else pe_cnt <= 5'd0;
    end
end

reg [31:0] pixel_00, pixel_10, pixel_20;
always @* begin
    if (pe_cnt <= 5'd3) begin
        pixel_00 = pixel_in0;
        pixel_10 = pixel_in1;
        pixel_20 = pixel_in2;    
    end
    else begin
        if (~pe_cnt[1]) begin // 00 or 01
            pixel_00 = nxt_in0;
            pixel_10 = nxt_in1;
            pixel_20 = nxt_in2; 
        end
        else begin
            pixel_00 = pixel_in0;
            pixel_10 = pixel_in1;
            pixel_20 = pixel_in2;         
        end
    end
end

always @* begin
    case (pe_cnt)
        6'd0, 6'd1, 6'd2, 6'd3, 6'd4, 6'd5, 6'd6, 6'd7, 6'd8, 6'd9, 
        6'd10, 6'd11, 6'd12, 6'd13, 6'd14, 6'd15: begin
            kernel_00 = k00; kernel_01 = k01; 
            kernel_10 = k10; kernel_11 = k11; 
            kernel_20 = k20; kernel_21 = k21; 
        end
        6'd16: begin
            kernel_00 = nxt_k00; kernel_01 = k01;
            kernel_10 = nxt_k10; kernel_11 = k11; 
            kernel_20 = nxt_k20; kernel_21 = k21;
        end
        6'd17: begin
            kernel_00 = nxt_k00; kernel_01 = nxt_k01;
            kernel_10 = nxt_k10; kernel_11 = nxt_k11;
            kernel_20 = nxt_k20; kernel_21 = nxt_k21;
        end
        default: begin
            kernel_00 = 32'd0; kernel_01 = 32'd0;
            kernel_10 = 32'd0; kernel_11 = 32'd0;
            kernel_20 = 32'd0; kernel_21 = 32'd0;
        end
    endcase
end


reg [31:0] pixel_01, pixel_11, pixel_21;
always @* begin
    if (pe_cnt <= 5'd3) begin
        pixel_01 = pixel_in0;
        pixel_11 = pixel_in1;
        pixel_21 = pixel_in2;    
    end
    else begin
        if (pe_cnt[1:0] == 2'b01) begin
            pixel_01 = nxt_in0;
            pixel_11 = nxt_in1;
            pixel_21 = nxt_in2; 
        end
        else begin
            pixel_01 = pixel_in0;
            pixel_11 = pixel_in1;
            pixel_21 = pixel_in2;         
        end
    end
end



always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        p00 <= 32'd0; p01 <= 32'd0; p02 <= 32'd0;
        p10 <= 32'd0; p11 <= 32'd0; p12 <= 32'd0;
        p20 <= 32'd0; p21 <= 32'd0; p22 <= 32'd0;
    end
    else begin
        p00 <= bias_L1[0]; p01 <= bias_L2[0]; p02 <= bias_L3[0];
        p10 <= bias_L1[1]; p11 <= bias_L2[1]; p12 <= bias_L3[1];
        p20 <= bias_L1[2]; p21 <= bias_L2[2]; p22 <= bias_L3[2];
    end
end

wire [31:0] PE01_out, PE11_out, PE21_out;
DW_fp_mult_inst PE00( .pixel(pixel_00), .kernel(kernel_00), .out(bias_L1[0]) );
DW_fp_mult_inst PE10( .pixel(pixel_10), .kernel(kernel_10), .out(bias_L1[1]) );
DW_fp_mult_inst PE20( .pixel(pixel_20), .kernel(kernel_20), .out(bias_L1[2]) );

DW_fp_mult_inst PE01( .pixel(pixel_01), .kernel(kernel_01), .out(PE01_out) );
DW_fp_mult_inst PE11( .pixel(pixel_11), .kernel(kernel_11), .out(PE11_out) );
DW_fp_mult_inst PE21( .pixel(pixel_21), .kernel(kernel_21), .out(PE21_out) );

DW_fp_add_inst PE001( .inst_a(PE01_out), .inst_b(p00), .z_inst(bias_L2[0]) );
DW_fp_add_inst PE011( .inst_a(PE11_out), .inst_b(p10), .z_inst(bias_L2[1]) );
DW_fp_add_inst PE021( .inst_a(PE21_out), .inst_b(p20), .z_inst(bias_L2[2]) );



DW_fp_mac_inst PE02( .pixel(pixel_in0), .kernel(k02), .bias(p01), .out(bias_L3[0]) );
DW_fp_mac_inst PE12( .pixel(pixel_in1), .kernel(k12), .bias(p11), .out(bias_L3[1]) );
DW_fp_mac_inst PE22( .pixel(pixel_in2), .kernel(k22), .bias(p21), .out(bias_L3[2]) );

DW_fp_sum3_inst sum3( .a(p02), .b(p12), .c(p22), .d(sum) );

endmodule


module out_convolution(
    clk, rst_n, ready, reset_signal,
    sum, conv_out, feature_cnt
);
input ready, reset_signal;
input clk, rst_n;
input [31:0] sum;
output [31:0] conv_out;
output reg [5:0] feature_cnt;

reg [31:0] c00_0, c01_0, c02_0, c03_0;
reg [31:0] c10_0, c11_0, c12_0, c13_0;
reg [31:0] c20_0, c21_0, c22_0, c23_0;
reg [31:0] c30_0, c31_0, c32_0, c33_0;

reg [31:0] c00_1, c01_1, c02_1, c03_1;
reg [31:0] c10_1, c11_1, c12_1, c13_1;
reg [31:0] c20_1, c21_1, c22_1, c23_1;
reg [31:0] c30_1, c31_1, c32_1, c33_1;

always @(posedge clk, negedge rst_n) begin
    if(!rst_n) feature_cnt <= 0;
    else begin
        if(ready && feature_cnt < 50) feature_cnt <= feature_cnt + 1;
        else if(feature_cnt == 50 || reset_signal)    feature_cnt <= 0;    
        else                          feature_cnt <= feature_cnt;

    end

end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        c00_0 <= 32'd0; c01_0 <= 32'd0; c02_0 <= 32'd0; c03_0 <= 32'd0;
        c10_0 <= 32'd0; c11_0 <= 32'd0; c12_0 <= 32'd0; c13_0 <= 32'd0;
        c20_0 <= 32'd0; c21_0 <= 32'd0; c22_0 <= 32'd0; c23_0 <= 32'd0;
        c30_0 <= 32'd0; c31_0 <= 32'd0; c32_0 <= 32'd0; c33_0 <= 32'd0;
        
        c00_1 <= 32'd0; c01_1 <= 32'd0; c02_1 <= 32'd0; c03_1 <= 32'd0;
        c10_1 <= 32'd0; c11_1 <= 32'd0; c12_1 <= 32'd0; c13_1 <= 32'd0;
        c20_1 <= 32'd0; c21_1 <= 32'd0; c22_1 <= 32'd0; c23_1 <= 32'd0;
        c30_1 <= 32'd0; c31_1 <= 32'd0; c32_1 <= 32'd0; c33_1 <= 32'd0;
    end
    else begin
        c00_0 <= c01_0; c01_0 <= c02_0; c02_0 <= c03_0; c03_0 <= c10_0;
        c10_0 <= c11_0; c11_0 <= c12_0; c12_0 <= c13_0; c13_0 <= c20_0;
        c20_0 <= c21_0; c21_0 <= c22_0; c22_0 <= c23_0; c23_0 <= c30_0;
        c30_0 <= c31_0; c31_0 <= c32_0; c32_0 <= c33_0; c33_0 <= c00_1;
        
        c00_1 <= c01_1; c01_1 <= c02_1; c02_1 <= c03_1; c03_1 <= c10_1;
        c10_1 <= c11_1; c11_1 <= c12_1; c12_1 <= c13_1; c13_1 <= c20_1;
        c20_1 <= c21_1; c21_1 <= c22_1; c22_1 <= c23_1; c23_1 <= c30_1;
        c30_1 <= c31_1; c31_1 <= c32_1; c32_1 <= c33_1; c33_1 <= sum;
    end
end

DW_fp_sum3_inst sum_conv( .a(c00_0), .b(c00_1), .c(sum), .d(conv_out) );

endmodule

module Max_pooling (
    clk, rst_n, feature_cnt, conv_out, reset_signal,
    out_pooling, pooling_ready
);
input clk, rst_n, reset_signal;
input [31:0] conv_out;
input [5:0] feature_cnt;
output  [31:0] out_pooling;
output reg pooling_ready;

reg [31:0] a, b, c, d, e, f;
wire [31:0] big1, big2, small1, small2, small3, biggest;

always @(posedge clk, negedge rst_n) begin
    if(!rst_n) begin
        a <= 0;
        b <= 0;
        c <= 0;
        d <= 0;
        e <= 0;
        f <= 0;
    end
    else begin
        case(feature_cnt)
            8'd35: a <= conv_out;
            8'd36: b <= conv_out;
            8'd37: e <= conv_out;
            8'd38: f <= conv_out;
            8'd39: c <= conv_out;
            8'd40: d <= conv_out;
            8'd41: begin
                a <= e;
                b <= f;
                c <= conv_out;
            end
            8'd42: d <= conv_out;
            8'd43: a <= conv_out;
            8'd44: b <= conv_out;
            8'd45: e <= conv_out;
            8'd46: f <= conv_out;
            8'd47: c <= conv_out;
            8'd48: d <= conv_out;
            8'd49: begin
                a <= e;
                b <= f;
                c <= conv_out;
            end
            8'd50: d <= conv_out;
        endcase
    end
end


always @(posedge clk, negedge rst_n) begin
    if(!rst_n) pooling_ready <= 0;
    else begin
        if(feature_cnt == 40)     pooling_ready <= 1;
        else if(feature_cnt == 0) pooling_ready <= 0;
        else                      pooling_ready <= pooling_ready;
    end
    
                       
end

DW_fp_cmp_inst dw1(.inst_a(a), .inst_b(b), .z0_inst(small1), .z1_inst(big1));
DW_fp_cmp_inst dw2(.inst_a(c), .inst_b(d), .z0_inst(small2), .z1_inst(big2));
DW_fp_cmp_inst dw3(.inst_a(big1), .inst_b(big2), .z0_inst(small2), .z1_inst(out_pooling));

endmodule


module FC_normal(
    clk, rst_n, weight, out_pooling, pooling_ready, input_cnt, in_valid, reset_signal,
    out_fc, fc_cnt
);
input clk, rst_n, pooling_ready, in_valid, reset_signal;
input [6:0] input_cnt;
input [31:0] out_pooling, weight;
output [31:0] out_fc;
output reg [3:0] fc_cnt;

reg [31:0] out_pooling_reg;
always @(posedge clk,negedge rst_n) begin
    if(!rst_n) out_pooling_reg <= 0;
    else       out_pooling_reg <= out_pooling;
end

//存weight
reg [31:0] w11, w12, w21, w22;
always @(posedge clk,negedge rst_n) begin
    if(!rst_n) begin
        w11 <= 0;
        w12 <= 0;
        w21 <= 0;
        w22 <= 0;
    end
    else begin
        if(in_valid) begin
            case(input_cnt)
                7'd0: w11 <= weight;
                7'd1: w12 <= weight;
                7'd2: w21 <= weight;
                7'd3: w22 <= weight;
                default: begin
                      w11 <= w11;
                      w12 <= w12;
                      w21 <= w21;
                      w22 <= w22;
                end
            endcase
        end
        else begin
            w11 <= w11;
            w12 <= w12;
            w21 <= w21;
            w22 <= w22;
        end
    end
end
/////


always @ (posedge clk, negedge rst_n) begin
    if(!rst_n) fc_cnt <= 1;
    else begin
        if(reset_signal)       fc_cnt <= 1;
        else if(pooling_ready) fc_cnt <= fc_cnt + 1;
        else                   fc_cnt <= 1;
    end
end

wire [31:0] mult1;
assign mult1 = ((fc_cnt == 2) || (fc_cnt == 4) || (fc_cnt == 10) || (fc_cnt == 12)) ? out_pooling_reg : out_pooling;

reg [31:0] mult2;
always @* begin
    case(fc_cnt)
        5'd1: mult2 = w11;
        5'd2: mult2 = w12;
        5'd3: mult2 = w21;
        5'd4: mult2 = w22;
        5'd9: mult2 = w11;
        5'd10: mult2 = w12;
        5'd11: mult2 = w21;
        5'd12: mult2 = w22;
        default : mult2 = 0;
    endcase
end



reg [31:0] mult_temp1, mult_temp2;
wire [31:0] adder_input;
wire [31:0] mult_out;

always @(posedge clk, negedge rst_n) begin
    if(!rst_n) begin
        mult_temp1 <= 0;
        mult_temp2 <= 0;
    end
    else begin
        if((fc_cnt == 1) || (fc_cnt == 3) || (fc_cnt == 9) || (fc_cnt == 11)) mult_temp1 <= mult_out;
        else                                                                  mult_temp1 <= mult_temp1;

        if((fc_cnt == 2) || (fc_cnt == 4) || (fc_cnt == 10) || (fc_cnt == 12)) mult_temp2 <= mult_out;
        else                                                                   mult_temp2 <= mult_temp2;
    end
end

assign adder_input = ((fc_cnt == 3) || (fc_cnt == 11)) ? mult_temp1 : mult_temp2;

wire [2:0] inst_rnd = 3'b000;
wire [7:0] status_inst;
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;



DW_fp_mult_inst dw11( .pixel(mult1), .kernel(mult2), .out(mult_out));
DW_fp_add_inst  dw22( .inst_a(mult_out), .inst_b(adder_input), .z_inst(out_fc));

endmodule 


module Normalization(
    clk, rst_n, reset_signal, out_fc, fc_cnt, 
    biggest, smallest, min_max_output, norm_ready
);

input clk, rst_n, reset_signal;
input [31:0] out_fc;
input [3:0] fc_cnt;
output [31:0] biggest, smallest;
output reg  norm_ready;
output reg [31:0]min_max_output;

reg [31:0] F1, F2, F3, F4;

always @ (posedge clk, negedge rst_n)begin
    if(!rst_n) norm_ready <= 0;
    else begin
        if(reset_signal) norm_ready <= 0;
        else if(fc_cnt == 12) norm_ready <= 1;
        else             norm_ready <= norm_ready;
    end
end




wire [31:0] f1;
assign f1 = (fc_cnt == 3) ? out_fc : 0;

always @(posedge clk, negedge rst_n) begin
    if(!rst_n) begin
        F1 <= 0;
        F2 <= 0;
        F3 <= 0;
        F4 <= 0;
    end
    else begin
        case(fc_cnt)
            5'd3:  F1 <= out_fc;
            5'd4:  F2 <= out_fc;
            5'd11: F3 <= out_fc;
            5'd12: F4 <= out_fc;
            default : begin
                F1 <= F1;
                F2 <= F2;
                F3 <= F3;
                F4 <= F4;
            end
        endcase
    end
end

reg [31:0] a_big_input, a_small_input;


always @ (posedge clk, negedge rst_n) begin
    if(!rst_n) begin
         a_big_input <= 0;
         a_small_input <= 0;
    end
    else begin
        if(fc_cnt == 5'd3 ) begin
            a_big_input <= out_fc;
            a_small_input <= out_fc;
        end
        else if((fc_cnt == 5'd4) || (fc_cnt == 5'd11) ||(fc_cnt == 5'd12)) begin
            a_big_input <= biggest;
            a_small_input <= smallest;
        end
        else begin
            a_big_input <= a_big_input;
            a_small_input <= a_small_input;
        end
    end
end


reg [2:0] cnt;

always @ (posedge clk, negedge rst_n) begin
    if(!rst_n) cnt <= 0;
    else begin
        if(norm_ready) cnt <= cnt + 1;
        else           cnt <= 0;
    end
end

always @* begin
    if(norm_ready) begin
        if(cnt == 0) begin
            min_max_output = F1;
        end
        else if(cnt == 1) begin
           min_max_output = F2;
        end
        else if(cnt == 2) begin
           min_max_output = F3;
        end
        else if(cnt == 3) begin
           min_max_output = F4;
        end
        else begin
            min_max_output = 0;
        end
    end
    else begin
        min_max_output = F1;
    end
end

wire [31:0] z00, z11;
DW_fp_cmp_inst cmpBIG( .inst_a(a_big_input), .inst_b(out_fc), .z0_inst(z00), .z1_inst(biggest));
DW_fp_cmp_inst cmpSMALL( .inst_a(a_small_input), .inst_b(out_fc), .z0_inst(smallest), .z1_inst(z11));

endmodule


module Activation(
    clk, rst_n, biggest, smallest, min_max_output, norm_ready, s_opt, fc_cnt,
    act_ready, act_out
);
input [3:0] fc_cnt;
input clk, rst_n, norm_ready;
input [1:0] s_opt;
input [31:0] biggest, smallest, min_max_output;
output reg act_ready;
output reg [31:0] act_out;

reg [31:0] Max, Min;

always @ (posedge clk, negedge rst_n) begin
    if(!rst_n) begin
        Max <= 0;
        Min <= 0;
    end
    else begin
        if(fc_cnt == 12) begin
            Max <= biggest;
            Min <= smallest;
        end
    end
end


reg [3:0] cnt;
always @(posedge clk, negedge rst_n) begin
    if(!rst_n) cnt <= 0;
    else begin
        if(norm_ready) begin
            if(cnt < 10) cnt <= cnt + 1;
            else         cnt <= 0;
        end
        else           cnt <= 0;
    end
end

wire [31:0] out_exp, out_exp_neg;
reg [31:0] pos_exp_reg1, neg_exp_reg1, pos_exp_reg2, neg_exp_reg2;
always @(posedge clk, negedge rst_n) begin
    if(!rst_n) begin
        pos_exp_reg1 <= 0;
        neg_exp_reg1 <= 0;
    end
    else begin
        pos_exp_reg1 <= out_exp;
        neg_exp_reg1 <= out_exp_neg;
    end
end

always @(posedge clk, negedge rst_n) begin
    if(!rst_n) begin
        pos_exp_reg2 <= 0;
        neg_exp_reg2 <= 0;
    end
    else begin
        pos_exp_reg2 <= pos_exp_reg1;
        neg_exp_reg2 <= neg_exp_reg1;
    end
end

reg [31:0] plus_1_1, plus_1_2, plus_2_1, plus_2_2;
always @* begin
        case(cnt)
            5'd0, 5'd1, 5'd2, 5'd3: begin
                plus_1_1 = min_max_output;
                plus_1_2 = {!Min[31], Min[30:0]};
                plus_2_1 = Max;
                plus_2_2 = {!Min[31], Min[30:0]};
            end
            5'd4, 5'd5, 5'd6, 5'd7: begin
                case(s_opt)
                    2'd0: begin
                        plus_1_1 = min_max_output;
                        plus_1_2 = Min;
                        plus_2_1 = Max;
                        plus_2_2 = Min;
                    end
                    2'd1: begin
                        plus_1_1 = pos_exp_reg2;
                        plus_1_2 = {!neg_exp_reg2[31], neg_exp_reg2[30:0]};
                        plus_2_1 = pos_exp_reg2;
                        plus_2_2 = neg_exp_reg2;
                    end
                    2'd2: begin
                        plus_1_1 = 32'h3f800000;
                        plus_1_2 = 0;
                        plus_2_1 = 32'h3f800000;
                        plus_2_2 = neg_exp_reg2;
                    end
                    2'd3: begin
                        plus_1_1 = 0;
                        plus_1_2 = 0;
                        plus_2_1 = 32'h3f800000;
                        plus_2_2 = pos_exp_reg2;
                    end
                endcase
                
            end
            default: begin
                plus_1_1 = 0;
                plus_1_2 = 0;
                plus_2_1 = 0;
                plus_2_2 = 0;
            end
        endcase
end

wire [31:0] out_plus1, out_plus2;
reg [31:0] out_plus1_reg, out_plus2_reg;
always @(posedge clk, negedge rst_n) begin
    if(!rst_n) begin
        out_plus1_reg <= 0;
        out_plus2_reg <= 0;
    end
    else begin
        out_plus1_reg <= out_plus1;
        out_plus2_reg <= out_plus2;
    end
end



wire [31:0] out_division;
reg [31:0] out_division_reg;
always @ (posedge clk, negedge rst_n) begin
    if(!rst_n) begin
        out_division_reg <= 0;
    end
    else begin
        out_division_reg <= out_division;
    end
end

wire [31:0] out_ln;
reg [31:0] out_ln_reg;
always @ (posedge clk, negedge rst_n) begin
    if(!rst_n) out_ln_reg <= 0;
    else       out_ln_reg <= out_ln;
end



always @* begin
    if(s_opt == 0 && (cnt > 1 && cnt < 6)) begin
        act_ready = 1;
        act_out = (out_division_reg[31]) ? {!out_division_reg[31], out_division_reg[30:0]} : out_division_reg;
    end
    else if(s_opt !== 0 && (cnt > 5 && cnt < 10)) begin
        act_ready = 1;
        act_out = (s_opt == 3) ? out_ln_reg : out_division_reg ;
    end
    else begin
        act_ready = 0;
        act_out = 0;
    end
end


DW_fp_add_inst dwadd1( .inst_a(plus_1_1), .inst_b(plus_1_2), .z_inst(out_plus1) );
DW_fp_add_inst dwadd2( .inst_a(plus_2_1), .inst_b(plus_2_2), .z_inst(out_plus2) );


DW_fp_div_inst dwdiv1( .inst_a(out_plus1_reg), .inst_b(out_plus2_reg), .z_inst(out_division) ); // a/b
DW_fp_ln_inst dwln( .inst_a(out_plus2_reg), .z_inst(out_ln) );


wire [7:0] status_inst;
DW_fp_exp_inst dwexp1( .inst_a(out_division_reg), .z_inst(out_exp));
DW_fp_exp_inst dwexp2( .inst_a({!out_division_reg[31], out_division_reg[30:0]}), .z_inst(out_exp_neg));
endmodule





// z = a*b + c*d
module DW_fp_dp2_inst( inst_a, inst_b, inst_c, inst_d, z_inst );
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
parameter inst_arch_type = 0;
input [inst_sig_width+inst_exp_width : 0] inst_a;
input [inst_sig_width+inst_exp_width : 0] inst_b;
input [inst_sig_width+inst_exp_width : 0] inst_c;
input [inst_sig_width+inst_exp_width : 0] inst_d;
wire [2:0] inst_rnd = 3'b000;
output [inst_sig_width+inst_exp_width : 0] z_inst;
wire [7 : 0] status_inst;
// Instance of DW_fp_dp2
DW_fp_dp2 #(inst_sig_width, inst_exp_width, inst_ieee_compliance, inst_arch_type)
U1 (
.a(inst_a),
.b(inst_b),
.c(inst_c),
.d(inst_d),
.rnd(inst_rnd),
.z(z_inst),
.status(status_inst) );
endmodule

module DW_fp_add_inst( inst_a, inst_b, z_inst );
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
input [inst_sig_width+inst_exp_width : 0] inst_a;
input [inst_sig_width+inst_exp_width : 0] inst_b;
wire [2:0] inst_rnd = 3'b000;
output [inst_sig_width+inst_exp_width : 0] z_inst;
wire [7 : 0] status_inst;
// Instance of DW_fp_add
DW_fp_add #(inst_sig_width, inst_exp_width, inst_ieee_compliance)
U1 ( .a(inst_a), .b(inst_b), .rnd(inst_rnd), .z(z_inst), .status(status_inst) );
endmodule

module DW_fp_exp_inst( inst_a, z_inst );
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
parameter inst_arch = 0;
input [inst_sig_width+inst_exp_width : 0] inst_a;
output [inst_sig_width+inst_exp_width : 0] z_inst;
wire [7 : 0] status_inst;
// Instance of DW_fp_exp
DW_fp_exp #(inst_sig_width, inst_exp_width, inst_ieee_compliance, inst_arch) U1 (
.a(inst_a),
.z(z_inst),
.status(status_inst) );
endmodule

module DW_fp_sub_inst( inst_a, inst_b, z_inst );
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
input [inst_sig_width+inst_exp_width : 0] inst_a;
input [inst_sig_width+inst_exp_width : 0] inst_b;
wire [2:0] inst_rnd = 3'b000;
output [inst_sig_width+inst_exp_width : 0] z_inst;
wire [7 : 0] status_inst;
// Instance of DW_fp_sub
DW_fp_sub #(inst_sig_width, inst_exp_width, inst_ieee_compliance)
U1 ( .a(inst_a), .b(inst_b), .rnd(inst_rnd), .z(z_inst), .status(status_inst) );
endmodule

// z = a/b
module DW_fp_div_inst( inst_a, inst_b, z_inst );
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
parameter inst_faithful_round = 0;
input [inst_sig_width+inst_exp_width : 0] inst_a;
input [inst_sig_width+inst_exp_width : 0] inst_b;
wire [2:0] inst_rnd = 3'b000;
output [inst_sig_width+inst_exp_width : 0] z_inst;
wire [7 : 0] status_inst;
// Instance of DW_fp_div
DW_fp_div #(inst_sig_width, inst_exp_width, inst_ieee_compliance, inst_faithful_round) U1
( .a(inst_a), .b(inst_b), .rnd(inst_rnd), .z(z_inst), .status(status_inst)
);
endmodule

module DW_fp_cmp_inst( inst_a, inst_b, z0_inst, z1_inst);
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
input [inst_sig_width+inst_exp_width : 0] inst_a;
input [inst_sig_width+inst_exp_width : 0] inst_b;
wire inst_zctr = 1'b0; // the result: z0 < z1
wire aeqb_inst;
wire altb_inst;
wire agtb_inst;
wire unordered_inst;
output [inst_sig_width+inst_exp_width : 0] z0_inst;
output [inst_sig_width+inst_exp_width : 0] z1_inst;
wire [7 : 0] status0_inst;
wire [7 : 0] status1_inst;
// Instance of DW_fp_cmp

DW_fp_cmp #(inst_sig_width, inst_exp_width, inst_ieee_compliance) U1(
.a(inst_a), .b(inst_b), .zctr(inst_zctr), .aeqb(aeqb_inst), 
.altb(altb_inst), .agtb(agtb_inst), .unordered(unordered_inst), 
.z0(z0_inst), .z1(z1_inst), .status0(status0_inst), 
.status1(status1_inst) );
endmodule

// Instance of DW_fp_mac, return pixel*kernel + bias
module DW_fp_mac_inst( pixel, kernel, bias, out );
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
input [inst_sig_width+inst_exp_width : 0] pixel;
input [inst_sig_width+inst_exp_width : 0] kernel;
input [inst_sig_width+inst_exp_width : 0] bias;
output [inst_sig_width+inst_exp_width : 0] out;

wire [2:0] inst_rnd = 3'b000;
wire [7:0] status_inst;

DW_fp_mac #(inst_sig_width, inst_exp_width, inst_ieee_compliance) U1 (
    .a(pixel),
    .b(kernel),
    .c(bias),
    .rnd(inst_rnd),
    .z(out),
    .status(status_inst) 
);
endmodule

//  d = a + b + c
module DW_fp_sum3_inst( a, b, c, d );
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
parameter inst_arch_type = 0;
input [inst_sig_width+inst_exp_width : 0] a;
input [inst_sig_width+inst_exp_width : 0] b;
input [inst_sig_width+inst_exp_width : 0] c;
output [inst_sig_width+inst_exp_width : 0] d;
// Instance of DW_fp_sum3

wire [2:0] inst_rnd = 3'b000;
wire [7:0] status_inst;

DW_fp_sum3 #(inst_sig_width, inst_exp_width, inst_ieee_compliance, inst_arch_type) U1 (
.a(a), .b(b), .c(c), .rnd(inst_rnd), .z(d), .status(status_inst) );
endmodule

// c = a*b
module DW_fp_mult_inst( pixel, kernel, out );
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
input [inst_sig_width+inst_exp_width : 0] pixel;
input [inst_sig_width+inst_exp_width : 0] kernel;
output [inst_sig_width+inst_exp_width : 0] out;
// Instance of DW_fp_mult

wire [2:0] inst_rnd = 3'b000;
wire [7:0] status_inst;

DW_fp_mult #(inst_sig_width, inst_exp_width, inst_ieee_compliance) U1 ( 
.a(pixel), .b(kernel), .rnd(inst_rnd), .z(out), .status(status_inst) );
endmodule

module DW_fp_ln_inst( inst_a, z_inst );
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
parameter inst_arch = 0;
input [inst_sig_width+inst_exp_width : 0] inst_a;
output [inst_sig_width+inst_exp_width : 0] z_inst;
wire [7 : 0] status_inst;
// Instance of DW_fp_exp
DW_fp_ln #(inst_sig_width, inst_exp_width, inst_ieee_compliance, inst_arch) U1 (
.a(inst_a),
.z(z_inst),
.status(status_inst) );
endmodule

module DW_fp_recip_inst( inst_a, inst_rnd, z_inst, status_inst );
parameter inst_sig_width = 23;
parameter inst_exp_width = 8;
parameter inst_ieee_compliance = 0;
parameter inst_faithful_round = 1;

input [inst_sig_width+inst_exp_width : 0] inst_a;
input [2 : 0] inst_rnd;
output [inst_sig_width+inst_exp_width : 0] z_inst;
output [7 : 0] status_inst;

// Instance of DW_fp_recip
DW_fp_recip #(inst_sig_width, inst_exp_width, inst_ieee_compliance,
inst_faithful_round) U1 (
.a(inst_a),
.rnd(inst_rnd),
.z(z_inst),
.status(status_inst) );
endmodule