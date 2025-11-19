| function            | y   | y_rho | source_signals | data | A   | ldet | Q   | shape | proportions | scale | location | Lt  | z   | LLdetS | LL  | dLL | fp  | zfp | g   | kp  |
| ------------------- | --- | ----- | -------------- | ---- | --- | ---- | --- | ----- | ----------- | ----- | -------- | --- | --- | ------ | --- | --- | --- | --- | --- | --- |
| update_sources!     |     |       | w              | r    | r   |      |     |       |             |       |          |     |     |        |     |     |     |     |     |     |
| calculate_ldet!     |     |       |                |      | r   | w    |     |       |             |       |          |     |     |        |     |     |     |     |     |     |
| calculate_y!        | w   |       | r              |      |     |      |     |       |             | r     | r        |     |     |        |     |     |     |     |     |     |
| update_y_rho!       | r   | w     |                |      |     |      |     | r     |             |       |          |     |     |        |     |     |     |     |     |     |
| calculate_Q!        |     | r     |                |      |     |      | w   | r     | r           | r     |          |     |     |        |     |     |     |     |     |     |
| calculate_u_and_Lt! |     |       |                |      |     | r    | r   |       |             |       |          | w   | w   | r      |     |     |     |     |     |     |
| calculate_LL!       |     |       |                |      |     |      |     |       |             |       |          | r   |     |        | w   |     |     |     |     |     |
| calculate_DLL!      |     |       |                |      |     |      |     |       |             |       |          |     |     |        | r   | w   |     |     |     |     |
| calculate_lrate!    |     |       |                |      |     |      |     |       |             |       |          |     |     |        |     | r   |     |     |     |     |
| update_proportions  |     |       |                |      |     |      |     |       | w           |       |          |     | r   |        |     |     |     |     |     |     |
| ffun                | r   | r     |                |      |     |      |     | r     |             |       |          |     |     |        |     |     | w   |     |     |     |
| zfp                 |     |       |                |      |     |      |     |       |             |       |          |     | r   |        |     |     | r   | w   |     |     |
| update_g            |     |       |                |      |     |      |     |       |             | r     |          |     |     |        |     |     |     | r   | w   |     |
| update_kp           |     |       |                |      |     |      |     |       |             | r     |          |     |     |        |     |     | r   | r   |     | w   |
| update_location     | r   |       |                |      |     |      |     | r     |             | r     | w        |     |     |        |     |     |     | r   |     | r   |
| update_scale        | r   | r     |                |      |     |      |     | r     |             | w     |          |     | r   |        |     |     |     | r   |     |     |
| update_shape        |     | r     |                |      |     |      |     | w     |             |       |          |     | r   |        |     |     |     |     |     |     |
| newton_method       | r   |       | r              |      | w   |      |     |       | r           |       | r        |     | r   |        |     |     | r   |     | r   | r   |
| reparametrize       |     |       |                |      | r   |      |     |       |             | w     | w        |     |     |        |     |     |     |     |     |     |

-- updated --

| function            | y   | y_rho | source_signals | data | A   | shape | proportions | scale | location | Lt  | z   | LLdetS | LL  | dLL | fp  | zfp | g   | kp  |
| ------------------- | --- | ----- | -------------- | ---- | --- | ----- | ----------- | ----- | -------- | --- | --- | ------ | --- | --- | --- | --- | --- | --- |
| update_sources!     |     |       | w              | r    | r   |       |             |       |          |     |     |        |     |     |     |     |     |     |
| calculate_y!        | w   |       | r              |      |     |       |             | r     | r        |     |     |        |     |     |     |     |     |     |
| update_y_rho!       | r   | w     |                |      |     | r     |             |       |          |     |     |        |     |     |     |     |     |     |
| calculate_u_and_Lt! |     | r     |                |      | r   | r     | r           | r     |          | w   | w   | r      | w   |     |     |     |     |     |
| calculate_DLL!      |     |       |                |      |     |       |             |       |          |     |     |        | r   | w   |     |     |     |     |
| calculate_lrate!    |     |       |                |      |     |       |             |       |          |     |     |        |     | r   |     |     |     |     |
| update_proportions  |     |       |                |      |     |       | w           |       |          |     | r   |        |     |     |     |     |     |     |
| ffun                | r   | r     |                |      |     | r     |             |       |          |     |     |        |     |     | w   |     |     |     |
| zfp                 |     |       |                |      |     |       |             |       |          |     | r   |        |     |     | r   | w   |     |     |
| update_g            |     |       |                |      |     |       |             | r     |          |     |     |        |     |     |     | r   | w   |     |
| update_kp           |     |       |                |      |     |       |             | r     |          |     |     |        |     |     | r   | r   |     | w   |
| update_location     | r   |       |                |      |     | r     |             | r     | w        |     |     |        |     |     |     | r   |     | r   |
| update_scale        | r   | r     |                |      |     | r     |             | w     |          |     | r   |        |     |     |     | r   |     |     |
| update_shape        |     | r     |                |      |     | w     |             |       |          |     | r   |        |     |     |     |     |     |     |
| newton_method       | r   |       | r              |      | w   |       | r           |       | r        |     | r   |        |     |     | r   |     | r   | r   |
| reparametrize       |     |       |                |      | r   |       |             | w     | w        |     |     |        |     |     |     |     |     |     |
