import pandera as pa
from pandera.typing import Series
import datetime


class TransactionSchema(pa.SchemaModel):
    order_id: Series[int] = pa.Field(ge=0)
    status: Series[str] = pa.Field(isin=['approved', 'declined', 'chargeback'])
    email_anoni: Series[str] = pa.Field(str_matches=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
    billing_country_code: Series[str] = pa.Field(nullable=True, str_length={"min_value": 2, "max_value": 2})
    shipping_country_code: Series[str] = pa.Field(nullable=True, str_length={"min_value": 2, "max_value": 2})
    shipping_method: Series[str] = pa.Field(nullable=True)
    created_at: Series[datetime.date] = pa.Field()
    total_spent: Series[float] = pa.Field(ge=0)
    currency_code: Series[str] = pa.Field(nullable=True, str_length={"min_value": 3, "max_value": 3})
    gateway: Series[str] = pa.Field(nullable=True)
    V1_link: Series[int] = pa.Field(nullable=True)
    V2_distance: Series[float] = pa.Field(nullable=True)
    V3_distance: Series[float] = pa.Field(nullable=True)
    V4_our_age: Series[float] = pa.Field(ge=0, nullable=True)
    V5_merchant_age: Series[float] = pa.Field(ge=0, nullable=True)
    V6_avs_result: Series[str] = pa.Field(nullable=True)
    V7_bill_ship_name_match: Series[str] = pa.Field(nullable=True)
    V8_ip: Series[float] = pa.Field(nullable=True)
    V9_cookie: Series[float] = pa.Field(nullable=True)
    V10_cookie: Series[float] = pa.Field(nullable=True)
    V11_cookie: Series[float] = pa.Field(nullable=True)

    class Config:
        coerce = True  # Automatically coerce column types to match the defined schema
        drop_invalid_rows = True
