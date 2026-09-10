# الخطوط

ضع هنا ملفات `.ttf` للخط المستخدم في الترجمة المدمجة.

libass على ويندوز كثيراً ما يفشل في إيجاد خط بالاسم فقط، ويرجع إلى خط
افتراضي لا يحتوي تركيبة لام-ألف — فتظهر مربعات فارغة داخل الكلمات.
تمرير المجلد عبر `fontsdir` يحل المشكلة نهائياً ويجعل المشروع محمولاً.

## Cairo

نزّل من https://fonts.google.com/specimen/Cairo ثم انسخ من مجلد `static`:

    Cairo-Regular.ttf
    Cairo-Bold.ttf
    Cairo-SemiBold.ttf

استخدم ملفات `static/` وليس ملف الـ variable font الواحد
(`Cairo[slnt,wght].ttf`) — دعم libass للخطوط المتغيرة غير موثوق.

اسم الخط في `.env` يجب أن يطابق اسم العائلة الداخلي: `SUBTITLE_FONT=Cairo`
