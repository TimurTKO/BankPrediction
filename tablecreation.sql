create table test4.bank_classification
(
    age Int32,
    job String,
    marital String,
    education String,
    default String,
    housing String,
    loan String,
    contact String,
    month String,
    day_of_week String,
    duration Int32,
    campaign Int32,
    pdays Int32,
    previous Int32,
    poutcome String,
    `emp.var.rate` Float64,
    `cons.price.idx` Float64,
    `cons.conf.idx` Float64,
    euribor3m Float64,
    `nr.employed` Float64,
    y String,
    RowId Int64
) ENGINE = MergeTree()
ORDER BY RowId

