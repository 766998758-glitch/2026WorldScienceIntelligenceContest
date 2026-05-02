# 2026WorldScienceIntelligenceContest
4.30 本次版本: 气象聚合特征 + 噪声特征删除（为删除minute基础上的版本）
  - 移除噪声特征: minute(0.0002), dayofweek(0.0072), ghi_min(0.0000)
  - 气象数据处理: 424个.nc文件 → 日度聚合(空间平均后取5风速+4辐照度统计量)
  - 下一步: 进一步精简气象特征(保留ws_mean/ghi_afternoon等2-3个核心变量)

5.1  本次版本：
-无气象 + 阈值 1250
-精简气象版（ghi_mean+ghi_night）+ 阈值 2000

5.2 本次版本： 
-lightbgm+滞后特征0.647	12733
-lightbgm+滞后特征+气象数据0.594	10974
