224-232

no sacar el extra node en demand

segun esto distributors no tienen cost (225)
segun esto only factory has C

observation space -pipelie lenght

X <- inventory at the beginning of each period (for each main node)
Y <- inventory pipeline at the beggining of each period (no se si son orders)
R <- replenishment orders at the beginning of each period
S <- units sold
D <- demand
U <- unfulfilled demand
P <- pofit at each node

action_log <- stores action (request order) for each main edge at each perido

when env created it already has the state(inventory, pipeline and demand)