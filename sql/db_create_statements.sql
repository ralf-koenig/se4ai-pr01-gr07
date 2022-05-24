CREATE SCHEMA IF NOT EXISTS language_identification;

CREATE TABLE IF NOT EXISTS language_identification.feedback
(
    id                         serial primary key,
    text_from_user_input       text not null,
    language_by_classifier     VARCHAR(10) not null,
    probability_by_classifier  numeric not null,
    language_suggested_by_user VARCHAR(10) not null,
    creation_timestamp         timestamp not null default current_timestamp
)
