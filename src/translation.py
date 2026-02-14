from typing import Dict

LABEL_MAP_ID: Dict[str, str] = {
    "cat": "kucing",
    "tabby": "kucing tabby",
    "tiger cat": "kucing belang",
    "Persian cat": "kucing persia",
    "Egyptian cat": "kucing mesir",
    "dog": "anjing",
    "golden retriever": "golden retriever",
    "Labrador retriever": "labrador retriever",
    "German shepherd": "german shepherd",
    "beagle": "beagle",
    "car wheel": "roda mobil",
    "sports car": "mobil sport",
    "ambulance": "ambulans",
    "fire engine": "mobil pemadam",
    "airliner": "pesawat penumpang",
    "banana": "pisang",
    "orange": "jeruk",
    "pizza": "pizza",
    "cheeseburger": "burger keju",
    "coffee mug": "cangkir kopi",
    "laptop": "laptop",
    "desktop computer": "komputer desktop",
    "cellular telephone": "ponsel",
    "television": "televisi",
    "bookshop": "toko buku",
}


def translate_label(label: str, language: str = "id") -> str:
    if language != "id":
        return label
    return LABEL_MAP_ID.get(label, label)
