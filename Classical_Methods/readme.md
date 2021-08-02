##### Flow : 흐름 (ex.시간당 몇대의 차량이 지나가는지)
##### Speed : 속도
##### Demand : 수요예측 (ex.택시)
##### Travel Time : 주행 시간
##### Occupancy : 점유율 (ex.도로길이와 존재하는 차량의 길이 합의 비율, 차량 감지 센서가 동작한 시간과 차량 감지한 시간의 비율 등)
<br>
<br>
<table>
    <thead>
        <tr>
            <th colspan=2>구분(Method)</th>
            <th>응용문제(Task)</th>
            <th>기법</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td colspan=2>통계 모델링</td>
            <td>Flow<br>Demend</td>
            <td>ARMA, ARIMA, 포아송 분포 모델</td>
        </tr>
        <tr>
            <td rowspan=3>기계학습 모델링</td>
            <td>속성 기반 모델</td>
            <td>Flow<br>Demand</td>
            <td>추가예정2</td>
        </tr>
        <tr>
            <td>가우시안 프로세스 모델</td>
            <td>Flow<br>Speed<br>Demand<br>Occupancy</td>
            <td>추가예정3</td>
        </tr>
        <tr>
            <td>상태 공간</td>
            <td>Flow<br>Speed<br>Demand<br>Travel Time<br>Occupancy</td>
            <td>추가예정4</td>
        </tr>
    </tbody>
</table>

## 1. 시계열 통계 모델링 설명 참조
    - https://m.blog.naver.com/bluefish850/220749045909
