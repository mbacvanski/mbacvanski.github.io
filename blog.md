---
title: Blog
layout: small_title
---

{% for post in site.posts %}
### [{{ post.title }}]({{ post.url }})

*{{ post.date | date: '%B %d, %Y' }}*

{{ post.content | strip_html | truncatewords: 40 }} [Read more]({{ post.url }})

{% endfor %}
