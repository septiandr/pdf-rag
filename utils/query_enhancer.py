class QueryEnhancer:

    def __init__(self):
        # Sinonim & konteks untuk hukum Indonesia
        self.context_map = {
            "uud": "Undang-Undang Dasar",
            "pancasila": "Pancasila dasar negara Indonesia",
            "presiden": "Presiden Republik Indonesia",
            "dpr": "Dewan Perwakilan Rakyat",
            "dpd": "Dewan Perwakilan Daerah",
            "ma": "Mahkamah Agung",
            "mk": "Mahkamah Konstitusi",
            "hak": "hak asasi manusia",
            "kewajiban": "kewajiban warga negara",
            "pasal": "pasal dalam undang-undang",
            "bab": "bagian dalam undang-undang",
            "silas": "sila dalam Pancasila",
        }

    def enhance(self, query: str) -> str:
        enhanced = query.lower()

        # Tambah konteks jika query terlalu pendek
        if len(query.split()) <= 3:
            for keyword, context in self.context_map.items():
                if keyword in enhanced:
                    enhanced = f"{query} {context}"
                    break

        # Jika query berupa pertanyaan, tambah konteks pencarian
        if enhanced.startswith(("apa ", "siapa ", "kapan ", "dimana ", "mengapa ", "bagaimana ")):
            enhanced = f"{enhanced} dalam hukum Indonesia"

        return enhanced
